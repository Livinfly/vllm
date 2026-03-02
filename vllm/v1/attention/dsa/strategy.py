from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass
import torch
import logging


if TYPE_CHECKING:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.v1.attention.dsa.config import DSAConfig, SelectionContext, SelectResult


logger = logging.getLogger(__name__)


class BaseSelectStrategy(ABC):

    def __init__(self, dsa_config: "DSAConfig"):
        self.dsa_config = dsa_config

    @abstractmethod
    def select(
        self,
        selection_context: "SelectionContext",
    ):
        raise NotImplementedError


class StaticSelectStrategy(BaseSelectStrategy):
    # TODO: static strategy, like streamingLLM
    pass


class QuestSelectStrategy(BaseSelectStrategy):

    def __init__(self, dsa_config: "DSAConfig"):
        super().__init__(dsa_config)
        # TODO: vllm_config.cache_config.block_size;
        # `check_and_update_config` for cuda, default 16, --block-size
        self.block_size = dsa_config.block_size
        print(f"DDSA: {__name__}, QuestSelectStrategy __init__", flush=True)
        # TODO: cached min, max blocks


    def _update_block_stats(self):
        # TODO: cached min / max blocks lazy_update
        pass

    def select(
        self,
        selection_context: "SelectionContext",
    ):
        """
        perfill:
        1. full
        2. select (TODO)

        per_head select (TODO)
        """
        from vllm.v1.attention.dsa.config import SelectResult

        # (varlen, num_q_heads, head_dim)
        query = selection_context.query
        # (num_blocks, block_size, num_kv_heads, head_dim)
        key_cache = selection_context.key_cache
        value_cache = selection_context.value_cache
        attn_metadata = selection_context.attn_metadata

        print(f"DDSA: {__name__}, QuestSelectStrategy select start", flush=True)
        # TODO: get_batch_size from attn_metadata, now for FA
        # assert isinstance(attn_metadata, "FlashAttentionMetadata")
        batch_size = attn_metadata.seq_lens.shape[0]
        device = query.device
        query_start_loc = attn_metadata.query_start_loc
        # (max_batch_size, max_num_blocks_per_req)   ; (batch_size, max_num_blocks_per_req)
        block_table = attn_metadata.block_table
        # (batch_size,)
        seq_lens = attn_metadata.seq_lens
        max_seq_len = attn_metadata.max_seq_len

        num_q_heads = query.shape[-2]
        head_dim = query.shape[-1]
        num_kv_heads = key_cache.shape[-2]

        print(f"DDSA: batch_size={batch_size}, "
              f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}", flush=True)
        print(f"DDSA: key_cache.shape={key_cache.shape}, query.shape={query.shape}", flush=True)

        # Prefill
        # if max_query_len > 1:
        #     print(f"DDSA: Prefill mode (max_query_len={max_query_len}), skip selection", flush=True)
        #     return key_cache, value_cache, None

        # Decode
        # print(f"DDSA: Decode mode (max_query_len=1), performing selection", flush=True)

        assert num_q_heads % num_kv_heads == 0, (
            f"num_q_heads ({num_q_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )
        group_size = num_q_heads // num_kv_heads

        if group_size > 1:
            key_cache_rep = torch.repeat_interleave(
                key_cache,
                group_size,
                dim=-2
            )
            # value_cache_rep = torch.repeat_interleave(value_cache, group_size, dim=-2)
        else:
            key_cache_rep = key_cache
            # value_cache_rep = value_cache

        # (batch_size, max_num_blocks_per_req)
        active_blocks_mask = (block_table > 0)

        # TODO: 1. min & max cached; 2. inf & -inf for empty posi.
        # (num_blocks, num_q_heads, head_dim)
        k_block_min = key_cache_rep.amin(dim=1)  
        k_block_max = key_cache_rep.amax(dim=1)

        print(f"DDSA: k_block_min.shape={k_block_min.shape}", flush=True)

        new_block_tables = []
        new_seq_lens_list = []

        for i in range(batch_size):
            q_start = query_start_loc[i]
            q_end = query_start_loc[i + 1]
            q_len = q_end - q_start
            # (query_len, num_q_heads, head_dim)
            q = query[q_start:q_end]

            active_mask = active_blocks_mask[i]
            all_req_blocks = block_table[i][active_mask]

            # 基于 seq_len 计算实际需要的 blocks num，避免使用 block_table 中可能的残留数据
            original_seq_len = seq_lens[i].item()
            actual_blocks_needed = (original_seq_len + self.block_size - 1) // self.block_size

            # 只取前 actual_blocks_needed 个 blocks
            req_blocks = all_req_blocks[:actual_blocks_needed]
            num_active_blocks = req_blocks.shape[0]

            print(f"DDSA: req {i}, seq_len={original_seq_len}, "
                  f"all_blocks={all_req_blocks.tolist()}, "
                  f"actual_needed={actual_blocks_needed}, "
                  f"used_blocks={req_blocks.tolist()}", flush=True)

            # prefill or ...
            if q_len > 1 or num_active_blocks == 0:
                new_block_tables.append(block_table[i:i+1])
                new_seq_lens_list.append(seq_lens[i:i+1])
                print(f"DDSA: prefill skip {i}")
                continue
            else:
                print(f"DDSA: decode {i}")

            # (num_active_blocks, num_q_heads, head_dim)
            req_k_min = k_block_min[req_blocks, :, :]
            req_k_max = k_block_max[req_blocks, :, :]

            # (query_len, num_q_heads, 1, head_dim)
            q_exp = q.unsqueeze(2)
            # req_k_min/max: (1, num_q_heads, num_active_blocks, head_dim)
            req_k_min_exp = req_k_min.permute(1, 0, 2).unsqueeze(0).contiguous()
            req_k_max_exp = req_k_max.permute(1, 0, 2).unsqueeze(0).contiguous()

            # (query_len, num_q_heads, num_active_blocks, head_dim)
            prod_min = q_exp * req_k_min_exp
            prod_max = q_exp * req_k_max_exp

            # (query_len, num_q_heads, num_active_blocks)
            scores_per_query = torch.maximum(prod_min, prod_max).sum(dim=-1)

            # (num_active_blocks,)
            block_scores = scores_per_query.sum(dim=(0, 1))

            print(f"DDSA: top_k {self.dsa_config.top_k}")
            top_k = min(self.dsa_config.top_k, num_active_blocks)
            _, top_k_indices = torch.topk(block_scores, k=top_k, largest=True)

            top_k_indices_sorted, _ = torch.sort(top_k_indices)
            print(f"DDSA: top_k_indices_sorted.shape {top_k_indices_sorted.shape}, _ {_}")
            selected_blocks = req_blocks[top_k_indices_sorted]
            print(f"DDSA: selected_blocks {selected_blocks}")

            max_num_blocks = block_table.shape[1]
            # (B, H, NB)
            new_block_table_row = torch.zeros(
                (max_num_blocks, ),
                dtype=block_table.dtype,
                device=device
            )
            print(f"DDSA: max_num_blocks {max_num_blocks}, top_k {top_k}, selected_blocks.shape {selected_blocks.shape}")
            new_block_table_row[:top_k] = selected_blocks
            new_block_tables.append(new_block_table_row.unsqueeze(0))

            original_seq_len = seq_lens[i].item()
            num_full_blocks = original_seq_len // self.block_size
            last_block_tokens = original_seq_len % self.block_size

            last_active_block = req_blocks[-1] if len(req_blocks) > 0 else None

            new_seq_len = (top_k-1) * self.block_size
            if (selected_blocks == last_active_block).any() and last_block_tokens > 0:
                new_seq_len += last_block_tokens
            else:
                new_seq_len += self.block_size

            new_seq_lens_list.append(torch.tensor([new_seq_len], dtype=seq_lens.dtype, device=device))

            print(f"DDSA: req {i}: original {num_active_blocks} blocks ({original_seq_len} tokens) -> "
                  f"selected {top_k} blocks ({new_seq_len} tokens)", flush=True)

        new_block_table = torch.cat(new_block_tables, dim=0)
        new_seq_lens = torch.cat(new_seq_lens_list, dim=0)
        new_max_seq_len = int(new_seq_lens.max().item())

        select_result = SelectResult(
            new_block_table=new_block_table,
            new_seq_lens=new_seq_lens,
            new_max_seq_len=new_max_seq_len,
            original_block_table=block_table.clone(),
            original_seq_lens=seq_lens.clone(),
            original_max_seq_len=max_seq_len,
        )

        print(f"DDSA: QuestSelectStrategy select end", flush=True)

        return key_cache, value_cache, select_result






class ClusterKVSelectStrategy(BaseSelectStrategy):
    pass


class DeepseekDSASelectStrategy(BaseSelectStrategy):
    pass


class DistributedDSASelectStrategy(BaseSelectStrategy):
    # TODO: DDSA
    pass


'''
backup
def select(
        self,
        selection_context: "SelectionContext",
    ):
        # (varlen, num_q_heads, head_dim)
        query = selection_context.query
        # (num_blocks, block_size, num_kv_heads, head_dim)
        key_cache = selection_context.key_cache
        value_cache = selection_context.value_cache
        attn_metadata = selection_context.attn_metadata
        print(f"DDSA: {__name__}, QuestSelectStrategy select, first test", flush=True)
        # TODO: get_batch_size from attn_metadata, now for FA
        # assert isinstance(attn_metadata, "FlashAttentionMetadata")
        batch_size, device = attn_metadata.seq_lens.shape[0], query.device  # type: ignore
        query_start_loc = attn_metadata.query_start_loc  # type: ignore


        # get block stats
        # (max_batch_size, max_num_blocks_per_req)   ; (batch_size, max_num_blocks_per_req)
        block_table = attn_metadata.block_table
        active_blocks_indices = (block_table > 0)
        # active_blocks = block_table[active_blocks_indices].unique()
        # active_blocks = block_table[active_blocks_indices]
        active_blocks = [
            block_table[i][active_blocks_indices[i]]
            for i in range(batch_size)
        ]

        # TODO: check
        # token_offs = torch.arange(self.block_size, device=key_cache.device)
        # token_posi = active_blocks + token_offs

        num_q_heads = query.shape[-2]
        head_dim = query.shape[-1]
        num_kv_heads = key_cache.shape[-2]

        # key_cache_active = key_cache[active_blocks, :, :, :]
        # value_cache_active = value_cache[active_blocks, :, :, :]

        print(f"DDSA: {__name__}, attn_metadata: ", flush=True)
        for k, v in attn_metadata.__dict__.items():
            print(f"DDSA: {__name__}, {k}: {v}", flush=True)

        print(f"DDSA: {__name__}, query.shape: {query.shape}", flush=True)
        print(f"DDSA: {__name__}, key_cache.shape: {key_cache.shape}", flush=True)

        print(f"DDSA: {__name__}, num_q_heads: {num_q_heads}, num_kv_heads: {num_kv_heads}", flush=True)
        # print(f"DDSA: {__name__}, block_table.shape: {block_table.shape}, active_blocks_indices.shape: {active_blocks_indices.shape}, active_blocks.shape: {active_blocks.shape}", flush=True)
        # print(f"DDSA: {__name__}, active_blocks: {active_blocks}", flush=True)
        # print(f"DDSA: {__name__}, block_table: {block_table}", flush=True)

        # print(f"DDSA: {__name__}, key_cache_active[:, :, 0, 0]: {key_cache_active[:, :, 0, 0]}", flush=True)
        # print(f"DDSA: {__name__}, value_cache_active[:, :, 0, 0]: {value_cache_active[:, :, 0, 0]}", flush=True)
    
        assert num_q_heads % num_kv_heads == 0, (
            f"num_q_heads ({num_q_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )

        group_size = num_q_heads // num_kv_heads
        if group_size > 1:
            key_cache_rep = torch.repeat_interleave(
                key_cache, 
                group_size,
                dim=-2
            )
            key_cache_rep = torch.repeat_interleave(
                value_cache, 
                group_size,
                dim=-2
            )

        return None, None, None
    
        k_block_min = torch.where(
            active_blocks_indices, 
            key_cache,
            torch.full_like(key_cache, float("inf"))
        ).amin(dim=-1)
        k_block_max = torch.where(
            active_blocks_indices, 
            key_cache,
            torch.full_like(key_cache, float("-inf"))
        ).amax(dim=-1)


        # for i in range(batch_size):
        #     k_block_min[i, active_blocks_indices[i, :]] = torch.amin(key_cache[active_blocks[i]], dim=-2)
        #     k_block_max[i, active_blocks_indices[i, :]] = torch.amax(key_cache[active_blocks[i]], dim=-2)

        print(f"DDSA: {__name__}, k_block_min.shape: {k_block_min.shape}", flush=True)
        print(f"DDSA: {__name__}, k_block_min: {k_block_min}", flush=True)
        print(f"DDSA: {__name__}, k_block_max.shape: {k_block_max.shape}", flush=True)
        print(f"DDSA: {__name__}, k_block_max: {k_block_max}", flush=True)

        for i in range(batch_size):
            q = query[query_start_loc[i] : query_start_loc[i + 1]]
'''