# Silent Cross-Adapter Contamination in Fused MoE LoRA Kernels

How shared-expert fusion causes a Triton kernel to read the wrong adapter's weights during multi-LoRA serving, without crashing or logging.

**Finding:** [sgl-project/sglang#40240](https://github.com/sgl-project/sglang/pull/40240) | **Upstream fork bug:** [togethercomputer/xorl-sglang#37](https://github.com/togethercomputer/xorl-sglang/issues/37)

## The setup

Modern MoE models like DeepSeek-V3 and Qwen3 have two kinds of experts: a set of **routed experts** (selected per-token by a gating network) and one or more **shared experts** (always active). SGLang and other inference engines optionally *fuse* the shared expert into the routed set during dispatch, giving it a synthetic expert ID equal to `num_routed_experts`. This avoids a separate kernel launch.

LoRA adapters for MoE models carry per-expert low-rank weight matrices. The stacked weight tensor has shape:

```
lora_a_stacked: (max_loras, num_experts, max_lora_rank, K)
```

where `max_loras` is the number of concurrently loaded adapters and `num_experts` is the number of *routed* experts (the shared expert has no LoRA weights in this tensor).

## The bug

The fused MoE LoRA Triton kernel indexes this tensor using `(lora_id, expert_id)`. It guards against the `-1` sentinel (no expert assigned), but does not check the upper bound:

```python
expert_id = tl.load(expert_ids_ptr + pid_m)
if expert_id == -1:
    return

b_ptrs = (
    cur_b_ptr
    + lora_id * stride_bl      # jump to the right adapter
    + expert_id * stride_be    # jump to the right expert
    + ...
)
```

`num_experts` is passed into the kernel but is never read. When shared-expert fusion is active, `expert_id` can equal `num_experts` (the fused shared expert's synthetic ID), which is one past the last valid index.

## Why it doesn't crash: the stride arithmetic

This is the subtle part. For a 4D tensor `(max_loras, num_experts, max_lora_rank, K)`:

```
stride_bl = num_experts * max_lora_rank * K    (stride along dim 0)
stride_be = max_lora_rank * K                  (stride along dim 1)
```

Substituting `expert_id = num_experts`:

```
lora_id * stride_bl + num_experts * stride_be
= lora_id * (num_experts * max_lora_rank * K) + num_experts * (max_lora_rank * K)
= (lora_id + 1) * num_experts * max_lora_rank * K
= (lora_id + 1) * stride_bl
```

This is exactly the base address of **adapter `lora_id + 1`, expert 0**.

## Memory layout

```
lora_a_stacked (max_loras=2, num_experts=4, max_lora_rank=R, K):

Adapter 0                              Adapter 1
+----------+----------+----+----------+----------+----------+----+----------+
| expert 0 | expert 1 | .. | expert 3 | expert 0 | expert 1 | .. | expert 3 |
+----------+----------+----+----------+----------+----------+----+----------+
                                       ^
                                       |
                              lora_id=0, expert_id=4
                              lands HERE: adapter 1, expert 0
```

When adapter 0 gets routed to the fused shared expert (expert_id=4), the kernel reads adapter 1's expert-0 weights and blends them into adapter 0's output. No crash, no NaN, no warning. Just wrong results.

## The three failure modes

| `lora_id` | Behaviour |
|---|---|
| `< max_loras - 1` | Reads the **next adapter's** expert 0. Deterministic, silent, wrong output. |
| `== max_loras - 1` | Reads past the tensor allocation. Undefined behavior. |
| Any, with only one adapter loaded | Reads uninitialized memory or the next allocation's bytes. May produce NaN, zero, or garbage depending on dtype and allocator state. |

The first case is the most dangerous: in production multi-adapter serving, one user's adapter silently leaks into another user's output. The contamination is deterministic and reproducible, but invisible without bit-level verification.

## Verification

Tested on an RTX 5060 Ti (sm_120, torch 2.11, triton 3.6) with `max_loras=2`, `num_experts=4`, only adapter 0 loaded:

1. Route adapter 0 to `expert_id = num_experts` on the unpatched kernel: `max|out| = 179.907`
2. Run adapter 0 at expert 0 with adapter 1's expert-0 weights copied in: same `179.907`, `max|A - B| = 0.000000000`
3. With the bounds check added: same routing gives `0.0` (correctly skipped)
4. In-range experts: identical results before and after the fix

The stray read lands bit-for-bit on the next adapter's expert 0, exactly as the stride arithmetic predicts.

## The fix

One scalar comparison, evaluated before any memory traffic, alongside the existing sentinel check:

```python
expert_id = tl.load(expert_ids_ptr + pid_m)
if expert_id == -1:
    return
if expert_id >= num_experts:   # <-- added
    return
```

Zero measurable throughput impact.

## Why existing tests missed it

The test helper `assign_experts_to_tokens` builds routing with `torch.randperm(num_experts)[:top_k]`, so expert IDs are always in `[0, num_experts)`. The out-of-range ID only appears when shared-expert fusion assigns the synthetic ID, which no unit test exercises.

## How vLLM avoids this

vLLM's `fused_moe.py` takes a different architectural approach. When LoRA is active, it explicitly disables shared-expert fusion:

```python
# vllm/lora/layers/fused_moe.py, line 73
moe_kernel.impl.shared_experts = None
```

LoRA weight buffers are shaped by `local_num_experts` (routed only), so the out-of-range ID can never appear. This is a design-level mitigation rather than a kernel-level guard, but it sacrifices the throughput benefit of shared-expert fusion for all LoRA requests.

SGLang's fix preserves fusion (the shared expert's forward pass still happens through the fused dispatch) while correctly skipping the LoRA delta for it.

## The broader pattern

This class of bug, where a valid index in one subsystem becomes an out-of-bounds index in another subsystem's tensor, is common in fused GPU kernels. The fusion optimization changes the expert ID space (adding synthetic IDs), but the LoRA kernel's weight tensors were sized for the original space. The `-1` sentinel check was correctly inherited from the non-fused path, but no equivalent check was added for the new upper bound introduced by fusion.

Multi-adapter serving amplifies the impact: instead of reading garbage (which might NaN and get noticed), the stray pointer lands on valid weights belonging to a different user's adapter. The output is plausible, the loss is reasonable, and the contamination is invisible without targeted verification.

## Reproduction

See [`repro/`](repro/) for the standalone reproduction script used to verify the stride arithmetic and produce the measurements above.

## Timeline

- 2026-09-18: Found while reading the fused_moe_lora kernel path after seeing [xorl-sglang#37](https://github.com/togethercomputer/xorl-sglang/issues/37)
- 2026-09-18: Filed [sgl-project/sglang#40240](https://github.com/sgl-project/sglang/pull/40240) with fix + deterministic test
- 2026-09-29: Verified vLLM's fused_moe_lora_op.py is architecturally safe (shared_experts = None when LoRA active)
