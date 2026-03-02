"""handled by original backend"""

# TODO: attention pre_ / post_ pipeline function


from abc import ABC, abstractmethod

from gguf import Optional

from vllm.v1.attention.dsa.kvcache_selector import KVCacheSelector
from vllm.v1.attention.dsa.metadata_adaptor import AttentionMetadataAdaptor, FlashAttentionMetadataAdaptor
from vllm.v1.attention.dsa.config import SelectStrategyType, SelectionContext
from vllm.v1.attention.dsa.utils import DSAConfig, get_dsa_config

class AttentionAdaptor(ABC):

    def __init__(
        self,
        dsa_config: Optional[DSAConfig] = None,
    ):
        self.dsa_config = dsa_config or get_dsa_config()
        self.kvcache_selector = KVCacheSelector(dsa_config=dsa_config)
        self.metadata_adaptor = self._create_metadata_adaptor()
        # recover
        self.last_select_result = None
        self.last_attn_metadata = None
    
    @abstractmethod
    def _create_metadata_adaptor(self) -> AttentionMetadataAdaptor:
        raise NotImplementedError

    @abstractmethod
    def pre_attention(self):
        return NotImplementedError
    
    @abstractmethod
    def pre_forward(
        self,
        query,
        key_cache,
        value_cache,
        attn_metadata,
    ):
        raise NotImplementedError

    @abstractmethod
    def post_forward(self, attn_metadata):
        raise NotImplementedError
        

class FlashAttentionAdaptor(AttentionAdaptor):

    def _create_metadata_adaptor(self):
        return FlashAttentionMetadataAdaptor(dsa_config=self.dsa_config)

    def pre_attention(self):
        # before attention backend impl
        # TODO: select kvcache & modify metadata
        pass

    def pre_forward(
        self,
        query,
        key_cache,
        value_cache,
        attn_metadata,
    ):
        """
        Returns:
            selected_key_cache
            selected_value_cache
            adapted_metadata
            select_result
        """
        if not self.dsa_config.dsa_enabled or self.dsa_config.strategy is SelectStrategyType.FULL:
            self.last_select_result = None
            self.last_attn_metadata = None
            return key_cache, value_cache, attn_metadata, None

        selection_context = SelectionContext(query, key_cache, value_cache, attn_metadata)

        selected_key_cache, selected_value_cache, select_result = self.kvcache_selector.select(
            selection_context,
            self.dsa_config.strategy,
        )


        self.last_select_result = select_result
        self.last_attn_metadata = attn_metadata

        adapted_metadata = self.metadata_adaptor.adapt(attn_metadata, select_result)
        return selected_key_cache, selected_value_cache, adapted_metadata, select_result

    def post_forward(self, attn_metadata):
        if self.last_select_result is None:
            return attn_metadata

        recovered_metadata = self.metadata_adaptor.recover(attn_metadata, self.last_select_result)

        self.last_select_result = None
        self.last_attn_metadata = None

        return recovered_metadata