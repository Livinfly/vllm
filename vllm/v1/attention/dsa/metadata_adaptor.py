# TODO: attention metadata adaptor

from typing import Optional
from abc import ABC, abstractmethod
import copy

from vllm.v1.attention.dsa.config import DSAConfig


class AttentionMetadataAdaptor(ABC):

    def __init__(self, dsa_config: Optional[DSAConfig] = None):
        self.dsa_config = dsa_config

    @abstractmethod
    def adapt(self, attn_metadata, select_result):
        """
        根据 select_result 修改 attn_metadata
        Args:
            attn_metadata: 原始的 attention metadata
            select_result: SelectResult 对象，包含选择的块信息
        Returns:
            adapted_metadata: 修改后的 metadata
        """
        pass

    @abstractmethod
    def recover(self, attn_metadata, select_result):
        """
        恢复原始的 attn_metadata
        Args:
            attn_metadata: 当前（可能已修改）的 metadata
            select_result: SelectResult 对象，包含备份的原始信息
        Returns:
            recovered_metadata: 恢复后的 metadata
        """
        raise NotImplementedError


class FlashAttentionMetadataAdaptor(AttentionMetadataAdaptor):

    def adapt(self, attn_metadata, select_result):
        """
        修改 FlashAttentionMetadata 以使用选中的块
        保持原始 metadata 不变，返回新的 metadata
        """
        if select_result is None:
            return attn_metadata

        # 浅拷贝 metadata，只替换需要修改的字段
        adapted_metadata = copy.copy(attn_metadata)
        
        print(f"DDSA: FlashAttentionMetadataAdaptor.adapt:")
        print("ori:")
        print(f"block_table: {adapted_metadata.block_table}")
        print(f"seq_lens: {adapted_metadata.seq_lens}")
        print(f"max_seq_len: {adapted_metadata.max_seq_len}")
        print("new:")
        print(f"block_table: {select_result.new_block_table}")
        print(f"seq_lens: {select_result.new_seq_lens}")
        print(f"max_seq_len: {select_result.new_max_seq_len}")

        # 更新 block_table 和 seq_lens
        adapted_metadata.block_table = select_result.new_block_table
        adapted_metadata.seq_lens = select_result.new_seq_lens
        adapted_metadata.max_seq_len = select_result.new_max_seq_len

        print(f"DDSA: FlashAttentionMetadataAdaptor adapted:", flush=True)
        print(f"  original max_seq_len: {select_result.original_max_seq_len} "
              f"-> new max_seq_len: {select_result.new_max_seq_len}", flush=True)

        return adapted_metadata

    def recover(self, attn_metadata, select_result):
        """
        恢复 FlashAttentionMetadata 到原始状态
        """
        if select_result is None:
            return attn_metadata

        # 浅拷贝并恢复原始字段
        recovered_metadata = copy.copy(attn_metadata)

        # 恢复原始的 block_table 和 seq_lens
        recovered_metadata.block_table = select_result.original_block_table
        recovered_metadata.seq_lens = select_result.original_seq_lens
        recovered_metadata.max_seq_len = select_result.original_max_seq_len

        print(f"DDSA: FlashAttentionMetadataAdaptor recovered to original", flush=True)

        return recovered_metadata