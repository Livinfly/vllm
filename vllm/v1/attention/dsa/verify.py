"""
Verify DSA module KV cache selection correctness.
Manually extract KV data from selected blocks and compute attention.
"""

import torch
import torch.nn.functional as F


def manual_attention_with_selected_blocks(
    query: torch.Tensor,              # (total_tokens, num_q_heads, head_dim)
    key_cache: torch.Tensor,          # (num_blocks, block_size, num_kv_heads, head_dim)
    value_cache: torch.Tensor,        # (num_blocks, block_size, num_kv_heads, head_dim)
    block_table: torch.Tensor,        # (batch_size, max_num_blocks)
    seq_lens: torch.Tensor,           # (batch_size,)
    query_start_loc: torch.Tensor,    # (batch_size + 1,)
    selected_block_table: torch.Tensor,  # (batch_size, max_num_blocks) selected blocks
    selected_seq_lens: torch.Tensor,     # (batch_size,) selected seq lengths
    block_size: int = 16,
    scale: float = None,
) -> torch.Tensor:
    """
    Manual implementation: extract KV data from selected blocks and compute attention.

    Used to verify if modifying metadata is equivalent to direct data extraction.
    """
    batch_size = seq_lens.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim = query.shape[2]

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # GQA: each KV head corresponds to group_size Q heads
    assert num_q_heads % num_kv_heads == 0
    group_size = num_q_heads // num_kv_heads

    outputs = []

    for i in range(batch_size):
        # Get query for current request
        q_start = query_start_loc[i]
        q_end = query_start_loc[i + 1]
        q = query[q_start:q_end]  # (query_len, num_q_heads, head_dim)
        query_len = q.shape[0]

        # Get selected blocks
        selected_blocks = selected_block_table[i]
        selected_blocks = selected_blocks[selected_blocks > 0]  # filter padding zeros
        num_selected_blocks = selected_blocks.shape[0]
        selected_seq_len = selected_seq_lens[i].item()

        if num_selected_blocks == 0:
            # No blocks selected, return zero vector
            output = torch.zeros_like(q)
            outputs.append(output)
            continue

        # Manually extract selected blocks from cache
        # key_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        selected_key_blocks = key_cache[selected_blocks]  # (num_selected_blocks, block_size, num_kv_heads, head_dim)
        selected_value_blocks = value_cache[selected_blocks]

        # Flatten to contiguous sequence: (num_selected_blocks * block_size, num_kv_heads, head_dim)
        k = selected_key_blocks.reshape(-1, num_kv_heads, head_dim)
        v = selected_value_blocks.reshape(-1, num_kv_heads, head_dim)

        # Take only valid tokens (based on selected_seq_len)
        k = k[:selected_seq_len]  # (selected_seq_len, num_kv_heads, head_dim)
        v = v[:selected_seq_len]

        # GQA: expand KV heads to match Q heads
        if group_size > 1:
            k = torch.repeat_interleave(k, group_size, dim=1)  # (selected_seq_len, num_q_heads, head_dim)
            v = torch.repeat_interleave(v, group_size, dim=1)

        # Compute attention scores
        # q: (query_len, num_q_heads, head_dim)
        # k: (selected_seq_len, num_q_heads, head_dim)
        # scores: (query_len, num_q_heads, selected_seq_len)
        scores = torch.einsum('qhd,khd->qhk', q, k) * scale

        # Causal mask (no mask needed for decode with query_len=1)
        if query_len > 1:
            # Prefill stage, need causal mask
            mask = torch.triu(
                torch.ones(query_len, selected_seq_len, dtype=torch.bool, device=q.device),
                diagonal=selected_seq_len - query_len + 1
            )
            scores.masked_fill_(mask.unsqueeze(1), float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (query_len, num_q_heads, selected_seq_len)

        # Compute output
        # attn_weights: (query_len, num_q_heads, selected_seq_len)
        # v: (selected_seq_len, num_q_heads, head_dim)
        # output: (query_len, num_q_heads, head_dim)
        output = torch.einsum('qhk,khd->qhd', attn_weights, v)

        outputs.append(output)

    # Concatenate outputs for all requests
    return torch.cat(outputs, dim=0)


def verify_attention_equivalence(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    selected_block_table: torch.Tensor,
    selected_seq_lens: torch.Tensor,
    output_from_flash_attn: torch.Tensor,
    block_size: int = 16,
    scale: float = None,
    atol: float = 1e-2,
    rtol: float = 1e-5,
) -> bool:
    """
    Verify if Flash Attention (with modified metadata) output
    is equivalent to manually extracting data from selected blocks.

    Returns:
        bool: True if outputs are within tolerance
    """
    manual_output = manual_attention_with_selected_blocks(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        selected_block_table=selected_block_table,
        selected_seq_lens=selected_seq_lens,
        block_size=block_size,
        scale=scale,
    )

    # Compare outputs
    is_close = torch.allclose(
        output_from_flash_attn,
        manual_output,
        rtol=rtol,
        atol=atol,
    )

    if not is_close:
        diff = (output_from_flash_attn - manual_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"DDSA Verify: Outputs NOT equivalent!")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  Flash Attn output sample: {output_from_flash_attn[0, 0, :5]}")
        print(f"  Manual output sample: {manual_output[0, 0, :5]}")
    else:
        print(f"DDSA Verify: ✓ Outputs are equivalent (within rtol={rtol}, atol={atol})")

    return is_close
