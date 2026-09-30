"""
Verify the stride-aliasing arithmetic without needing a GPU or Triton.

Demonstrates that indexing a 4D tensor at (lora_id, num_experts) lands
exactly on (lora_id + 1, 0) due to contiguous stride layout.

Usage:
    python verify_stride_aliasing.py
"""

import struct


def main():
    max_loras = 2
    num_experts = 4
    max_lora_rank = 8
    K = 16

    shape = (max_loras, num_experts, max_lora_rank, K)
    numel = 1
    for s in shape:
        numel *= s

    # Fill each (lora_id, expert_id) block with a recognizable tag
    data = [0.0] * numel
    for lid in range(max_loras):
        for eid in range(num_experts):
            tag = lid * 100 + eid  # e.g. adapter 0 expert 3 -> 3.0
            base = (
                lid * num_experts * max_lora_rank * K
                + eid * max_lora_rank * K
            )
            for i in range(max_lora_rank * K):
                data[base + i] = float(tag)

    # Compute strides (contiguous, row-major)
    stride_bl = num_experts * max_lora_rank * K  # stride along dim 0
    stride_be = max_lora_rank * K                # stride along dim 1
    stride_bk = K                                # stride along dim 2
    stride_bn = 1                                # stride along dim 3

    print(f"Shape: {shape}")
    print(f"Strides: bl={stride_bl}, be={stride_be}, bk={stride_bk}, bn={stride_bn}")
    print()

    # The bug: lora_id=0, expert_id=num_experts (the fused shared expert)
    lora_id = 0
    expert_id = num_experts  # out of range!

    bug_offset = lora_id * stride_bl + expert_id * stride_be
    correct_offset = (lora_id + 1) * stride_bl + 0 * stride_be

    print(f"Out-of-range access: lora_id={lora_id}, expert_id={expert_id}")
    print(f"  Computed offset: {bug_offset}")
    print(f"  Adapter {lora_id+1} expert 0 offset: {correct_offset}")
    print(f"  Match: {bug_offset == correct_offset}")
    print(f"  Tag at bug offset: {data[bug_offset]}")
    print(f"  Expected (adapter 1, expert 0): {1 * 100 + 0}")
    print()

    # Verify for all adapters
    print("Full verification:")
    for lid in range(max_loras):
        bug_off = lid * stride_bl + num_experts * stride_be
        if bug_off < len(data):
            tag = data[bug_off]
            target_lid = int(tag) // 100
            target_eid = int(tag) % 100
            print(
                f"  adapter {lid}, expert {num_experts} -> "
                f"reads adapter {target_lid} expert {target_eid} "
                f"(tag={tag})"
            )
        else:
            print(
                f"  adapter {lid}, expert {num_experts} -> "
                f"OUT OF BOUNDS (offset {bug_off} >= {len(data)})"
            )

    print()
    print("Conclusion: the stray expert_id reads the next adapter's expert 0,")
    print("silently blending another user's LoRA weights into the output.")


if __name__ == "__main__":
    main()
