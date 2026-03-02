from turtle import st
from typing import Optional
import torch

from vllm.v1.attention.dsa.config import DSAConfig, SelectStrategyType, get_strategy_map, SelectionContext
from vllm.v1.attention.dsa.strategy import (
    BaseSelectStrategy,
    QuestSelectStrategy,
)
from vllm.v1.attention.dsa.utils import get_dsa_config

class KVCacheSelector:

    def __init__(
        self,
        dsa_config: Optional[DSAConfig] = None,
    ):
        self.dsa_config = dsa_config or get_dsa_config()

    def select(
        self,
        selection_context: SelectionContext,
        strategy_type: SelectStrategyType,
    ):
        # TODO: support different kvcache selection strategy for different attention layer
        # TODO: Lifecycle Management...
        strategy_map = get_strategy_map()
        strategy_impl_cls = strategy_map.get(strategy_type)
        assert strategy_impl_cls is not None
        strategy_impl = strategy_impl_cls(dsa_config=self.dsa_config)
        print(f"DDSA: {__name__}, KVCacheSelector, strategy_impl: {strategy_impl}", flush=True)

        selected_key_cache, selected_value_cache, select_result = \
            strategy_impl.select(selection_context=selection_context)

        return selected_key_cache, selected_value_cache, select_result