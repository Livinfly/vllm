"""
the definations of configs.
"""

from typing import Optional, List, Dict, Tuple, Union, TYPE_CHECKING, Any
from dataclasses import dataclass
from enum import Enum
import torch


@dataclass
class SelectResult:
    # new
    # (batch_size, max_num_blocks_per_req)
    new_block_table: torch.Tensor
    # (batch_size,)
    new_seq_lens: torch.Tensor
    new_max_seq_len: int

    # backup
    original_block_table: torch.Tensor
    original_seq_lens: torch.Tensor
    original_max_seq_len: int


class SelectStrategyType(Enum):
    FULL = "full"
    QUEST = "quest"
    CLUSTER_KV = "cluster_kv"
    DEEPSEEK_DSA = "dpsk_dsa"
    DISTRIBUTED_DSA = "ddsa"


def get_strategy_map():
    """延迟导入避免循环依赖"""
    from vllm.v1.attention.dsa.strategy import QuestSelectStrategy

    return {
        # SelectStrategyType.FULL: None,
        SelectStrategyType.QUEST: QuestSelectStrategy,
        # SelectStrategyType.CLUSTER_KV: None,
        # SelectStrategyType.DEEPSEEK_DSA: None,
        # SelectStrategyType.DISTRIBUTED_DSA: None,
    }


@dataclass
class SelectionContext:
    # from vllm.v1.attention.backend import AttentionMetadata

    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    attn_metadata: Any  # AttentionMetadata 类型在运行时可用


@dataclass
class DSAConfig:
    """
    Default configs for global DSA.
    """
    dsa_enabled: bool = False
    block_size: int = 16
    strategy: SelectStrategyType = SelectStrategyType.QUEST


    # TODO: custom config could be passed as args in `kvcache_select` / `per_layer_strategy` (int / str) -> strategy
    per_layer_strategy: Optional[Dict[str, SelectStrategyType]] = None

    # QUEST
    top_ratio: float = 0.5
    top_k: int = 3 # 32
    # TODO: top_ratio: Dict[int/str, float]
    # TODO
    use_cache_min_max_blocks: bool = False

    # CLUSTER_KV

    # DEEPSEEK_DSA

    # DISTRIBUTED_DSA
    distributed_enabled: bool = False

    # Others

    # verify
    verify_enabled: bool = True