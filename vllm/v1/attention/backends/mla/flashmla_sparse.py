# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.pcp import get_pcp_local_rows_for_range
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    CommonAttentionMetadata,
    MultipleOf,
    PCPQueryRoutingMetadata,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode,
    split_prefill_chunks,
)
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# For FP8 sparse attention we have two implementations:
# 1. Mixed batch mode: use the FP8 decode kernel for both prefill and decode this is
#    done by treating all tokens as single batch.
# 2. Separate prefill and decode mode: use the BF16 prefill kernel for prefill
#    (upconverting the FP8 cache to BF16 then calling the prefill kernel) and using
#    the FP8 decode kernel for decode.
# Currently we use #1 when the number of heads per rank is low (i.e. TP) since the BF16
# prefill kernel requires padding the number of heads to 128 while the decode does not
# so when the per-rank head count is below MIN_HEADS_FOR_BF16_PREFILL we use the mixed
# batch mode (#1).
MIN_HEADS_FOR_BF16_PREFILL = 32
FP8_DS_MLA_ENTRY_BYTES = 656

"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

For DeepSeek V3.2, in the "FP8 with scale" format, each token's KV cache is 656
Bytes, structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values.
    The first `float32` is the scale for the first 128 `float8_e4m3` values,
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.

For DeepSeek V4, in the "FP8 with scale" format, each token's KV cache is 584
Bytes, structured as:
-   **First 448 bytes:** The "quantized NoPE" part, containing 448
    `float8_e4m3` values.
-   **Next 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.
-   **Last 8 bytes:** Scale factors, containing 7 `ue8m0` values + 1B pad.
    The first `ue8m0` is the scale for the first 64 `float8_e4m3` values,
    the second for the next 64, and so on.
"""


class FlashMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]:
        return FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseImpl"]:
        return FlashMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V3.2 layout: 512 NoPE + 64 RoPE = 576.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def requires_pcp_query_routing(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # V3.2 main MLA: 656-byte custom storage format. See module docstring.
            return (num_blocks, block_size, 656)
        else:
            return (num_blocks, block_size, head_size)


@dataclass
class FlashMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048

    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    seq_lens: torch.Tensor | None = None
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    cp_kv_cache_interleave_size: int = 1

    @dataclass
    class FP8KernelMetadata:
        scheduler_metadata: FlashMLASchedMeta
        dummy_block_table: torch.Tensor
        cache_lens: torch.Tensor

    @dataclass
    class FP8SeparatePrefillDecode:
        @dataclass
        class Decode:
            seq_lens: torch.Tensor
            kernel_metadata: "FlashMLASparseMetadata.FP8KernelMetadata"
            decode_query_len: int  # needed for reshape in spec decode

        @dataclass
        class Prefill:
            # Request ID for each token: -1 for decode tokens, request index
            # (0, 1, 2, ...) for prefill tokens.
            # Shape: [num_actual_tokens]
            request_ids: torch.Tensor

            # Workspace start offsets for all prefill requests
            # Shape: [num_prefill_reqs], adjusted in-place per chunk to be
            # 0-indexed within each chunk. Used to map prefill tokens to workspace
            # offsets in convert_logical_index_to_physical_index
            workspace_starts: torch.Tensor

            @dataclass
            class Chunk:
                """Metadata for a chunk of prefill requests.

                Prefill requests may be chunked to fit within the fixed workspace size.
                """

                tokens_slice: slice
                block_table: torch.Tensor
                req_start_idx: int
                workspace_starts: torch.Tensor
                chunk_tot_seqlen: int
                local_cu_seq_lens: torch.Tensor | None = None
                dcp_workspace_starts: torch.Tensor | None = None
                dcp_owner_stride: int = 0
                local_total_seq_len: int = 0

            chunks: list[Chunk]

        num_prefills: int = 0
        num_decodes: int = 0
        num_prefill_tokens: int = 0
        num_decode_tokens: int = 0

        decode: Decode | None = None
        prefill: Prefill | None = None

    fp8_extra_metadata: FP8SeparatePrefillDecode | FP8KernelMetadata | None = None
    fp8_use_mixed_batch: bool = False
    pcp_query_routing: PCPQueryRoutingMetadata | None = None
    pcp_prefill_mode: str = "q_route"


@triton.jit
def _upconvert_fp8_ds_mla_workspace_kernel(
    src_ptr,
    dst_ptr,
    src_stride,
    dst_stride,
):
    token_id = tl.program_id(0)

    nope_offsets = tl.arange(0, 512)
    src_row = src_ptr + token_id * src_stride
    raw_nope = tl.load(src_row + nope_offsets)
    scale_offsets = 512 + (nope_offsets // 128) * 4
    scale_bits = (
        tl.load(src_row + scale_offsets).to(tl.uint32)
        | (tl.load(src_row + scale_offsets + 1).to(tl.uint32) << 8)
        | (tl.load(src_row + scale_offsets + 2).to(tl.uint32) << 16)
        | (tl.load(src_row + scale_offsets + 3).to(tl.uint32) << 24)
    )
    scales = scale_bits.to(tl.float32, bitcast=True)
    nope = raw_nope.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scales
    tl.store(dst_ptr + token_id * dst_stride + nope_offsets, nope)

    rope_offsets = tl.arange(0, 64)
    rope_bytes = src_row + 528 + rope_offsets * 2
    rope_bits = tl.load(rope_bytes).to(tl.uint16) | (
        tl.load(rope_bytes + 1).to(tl.uint16) << 8
    )
    rope = rope_bits.to(tl.bfloat16, bitcast=True)
    tl.store(dst_ptr + token_id * dst_stride + 512 + rope_offsets, rope)


def upconvert_fp8_ds_mla_workspace(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    """Upconvert packed contiguous fp8_ds_mla rows into a BF16 workspace."""
    assert src.ndim == 2 and src.shape[1] == FP8_DS_MLA_ENTRY_BYTES
    assert src.dtype == torch.uint8 and src.is_contiguous()
    assert dst.shape == (src.shape[0], 576)
    assert dst.dtype == torch.bfloat16 and dst.is_contiguous()
    if src.shape[0] == 0:
        return
    _upconvert_fp8_ds_mla_workspace_kernel[(src.shape[0],)](
        src,
        dst,
        src.stride(0),
        dst.stride(0),
        num_warps=8,
    )


def map_materialized_dcp_topk(
    topk_indices: torch.Tensor,
    request_ids: torch.Tensor,
    workspace_starts: torch.Tensor,
    owner_stride: int,
    dcp_world_size: int,
    cp_kv_cache_interleave_size: int,
    request_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map per-request global token IDs into rank-major owner workspaces."""
    safe_indices = topk_indices.clamp_min(0)
    owner = (safe_indices // cp_kv_cache_interleave_size) % dcp_world_size
    local_indices = (
        safe_indices // (dcp_world_size * cp_kv_cache_interleave_size)
    ) * cp_kv_cache_interleave_size + (safe_indices % cp_kv_cache_interleave_size)

    local_request_ids = request_ids.to(torch.int64) - request_offset
    valid_request = (local_request_ids >= 0) & (
        local_request_ids < workspace_starts.shape[0]
    )
    safe_request_ids = local_request_ids.clamp(0, workspace_starts.shape[0] - 1)
    starts = workspace_starts[safe_request_ids[:, None], owner.to(torch.int64)]
    mapped = owner * owner_stride + starts + local_indices
    valid = (topk_indices >= 0) & valid_request[:, None]
    mapped = torch.where(valid, mapped, -1).to(torch.int32)
    return mapped, valid.sum(dim=-1, dtype=torch.int32)


def get_prefill_workspace_size(max_model_len: int):
    # NOTE(Lucas): 5 is a magic number for controlling the prefill buffer size.
    # May be tuned later.
    # Memory usage: 5 * max_model_len * 576 * 2 bytes
    #   Example: DeepSeek-V3.2 with max_model_len=163840 ->
    #            5 * 163840 * 576 * 2 = ~900 MB
    # This fits nicely below the typical MoE workspace size of >2GB so this is "free"
    return max_model_len * 5


class FlashMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    require_uniform_decodes: ClassVar[bool] = True
    metadata_cls = FlashMLASparseMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        num_q_heads = self.model_config.get_num_attention_heads(parallel_config)
        if current_platform.is_device_capability_family(100):
            threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
                num_q_heads, 1024
            )
        else:
            threshold = {16: 128, 32: 128, 64: 256, 128: 256}.get(num_q_heads, 256)
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=(parallel_config.cp_kv_cache_interleave_size == 1),
        )

        sm_count = num_compute_units(device.index)

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        # FP8 decode kernel only supports h_q = 64 or 128, so we need to pad
        self.fp8_decode_padded_heads = (
            FlashMLASparseImpl._compute_fp8_decode_padded_heads(self.num_heads)
        )

        self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        # Shape: [max_num_seqs], all elements = topk_tokens (constant for full-CG)
        self.topk_tokens_tensor = torch.full(
            (max_num_seqs,), self.topk_tokens, device=device, dtype=torch.int32
        )
        # Shape: [max_num_seqs], all elements = max_model_len
        self.max_model_len_tensor = torch.full(
            (max_num_seqs,),
            self.model_config.max_model_len,
            device=device,
            dtype=torch.int32,
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (max_num_seqs, 1), dtype=torch.int32, device=self.device
        )

        # Equation taken from FlashMLA/csrc/api/sparse_decode.h
        # For sparse FP8 decode, the formula depends on architecture:
        # - SM90 (Hopper): num_sm_parts = num_sms / s_q / (h_q/64)
        # - SM100 (Blackwell head64/head64x2): num_sm_parts = num_sms / s_q
        # - SM100 (Blackwell head128): num_sm_parts = num_sms / s_q / 2
        # For max buffer size, use s_q = 1 (the case that produces largest output)
        # Use padded head count since that's what will be passed to the kernel
        h_q = self.fp8_decode_padded_heads
        if current_platform.is_device_capability_family(100):
            # SM100 head64 or head64x2 uses full SM count
            max_num_sm_parts = sm_count
        else:
            # SM90 uses h_q/64 divisor
            max_num_sm_parts = sm_count // max(1, h_q // 64)
        self.tile_scheduler_metadata_buffer = torch.empty(
            # TileSchedulerMetaDataSize = 8
            # see: FlashMLA/csrc/params.h
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device=device,
        )
        # Sized for per-request batching (num_decodes + 1)
        self.num_splits_buffer = torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        )

        self.pcp_dcp = (
            parallel_config.decode_context_parallel_size > 1
            and parallel_config.prefill_context_parallel_size > 1
        )
        self.pcp_prefill_mode = vllm_config.attention_config.sparse_mla_pcp_prefill_mode
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.fp8_default_mixed_batch = self.num_heads < MIN_HEADS_FOR_BF16_PREFILL

    def _build_fp8_mixed_decode_prefill(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> "FlashMLASparseMetadata.FP8KernelMetadata":
        """Build FP8 metadata treating MQA tokens as one batch.

        The scheduler initializes lazily from the runtime query shape, which may
        be the full batch or only decodes when prefills use dense MHA. This avoids
        the BF16 prefill kernel's head-padding overhead at high TP.
        """
        num_tokens = common_attn_metadata.num_actual_tokens

        # Use padded head count since that's what the kernel will see
        padded_heads = self.fp8_decode_padded_heads

        # Build metadata for all tokens as a single batch
        scheduler_metadata, _ = get_mla_metadata(
            cache_seqlens=self.topk_tokens_tensor[:1],  # Single batch
            num_q_tokens_per_head_k=num_tokens * padded_heads,
            topk=self.topk_tokens,
            num_heads_q=padded_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )

        fp8_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=scheduler_metadata,
            cache_lens=self.max_model_len_tensor[:1],
            dummy_block_table=self.dummy_block_table[:1],
        )

        return fp8_metadata

    def _build_fp8_separate_prefill_decode(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        metadata: FlashMLASparseMetadata,
    ) -> "FlashMLASparseMetadata.FP8SeparatePrefillDecode":
        num_tokens = common_attn_metadata.num_actual_tokens

        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            metadata.num_decodes,
            metadata.num_prefills,
            metadata.num_decode_tokens,
            num_tokens - metadata.num_decode_tokens,
        )

        decode_query_len = 0
        active_num_decodes = num_decodes
        if num_decodes > 0:
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            decode_query_len = (query_start_loc_cpu[1] - query_start_loc_cpu[0]).item()
            assert decode_query_len > 0
            active_num_decodes = num_decode_tokens // decode_query_len
            assert active_num_decodes * decode_query_len == num_decode_tokens

        FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
        fp8_metadata = FP8Meta(
            num_decodes=active_num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
        )

        # Extract prefill sequence lengths (context + query, not just query)
        # Decode requests come first in the batch, prefill requests follow
        prefill_request_id = None
        prefill_workspace_starts = None
        prefill_chunks = None

        # For pure decode batches, prefill_request_id will be None
        # For mixed batches, it will have -1 for decode and request_id for prefill
        if num_prefills > 0:
            # Upper bound is exact for prefill rows (the `[num_decodes:]`
            # slice below), so no D2H sync is needed.
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

            prefill_seq_lens_cpu = seq_lens_cpu[num_decodes:]
            materialize_dcp_kv = getattr(self, "pcp_dcp", False) and (
                self.pcp_prefill_mode == "kv_materialize"
            )
            dcp_seq_lens_cpu = None
            if materialize_dcp_kv:
                dcp_seq_lens_cpu = get_dcp_local_seq_lens(
                    prefill_seq_lens_cpu,
                    self.dcp_world_size,
                    None,
                    self.cp_kv_cache_interleave_size,
                )

            # Build prefill_request_id: -1 for decode, request index for
            # prefill. This enables a single
            # convert_logical_index_to_physical_index call for all tokens
            prefill_request_id = torch.full(
                (num_tokens,), -1, dtype=torch.int32, device=self.device
            )
            # Map prefill tokens to their request IDs (0, 1, 2, ...)
            for req_idx in range(num_prefills):
                # Get query token range for this prefill request
                global_req_idx = num_decodes + req_idx
                req_query_start = query_start_loc_cpu[global_req_idx]
                req_query_end = query_start_loc_cpu[global_req_idx + 1]
                prefill_request_id[req_query_start:req_query_end] = req_idx

            # will be adjusted by chunk loop
            prefill_workspace_starts_cpu = torch.zeros(
                num_prefills, dtype=torch.int32, pin_memory=True
            )
            prefill_workspace_starts_cpu[1:] = torch.cumsum(
                prefill_seq_lens_cpu[:-1], dim=0
            )
            # populated by non-blocking copy after prefill_workspace_starts_cpu is
            # updated by each chunk
            prefill_workspace_starts = torch.empty(
                num_prefills, dtype=torch.int32, device=self.device
            )

            # Chunk prefill requests to fit within workspace size
            max_prefill_buffer_size = get_prefill_workspace_size(
                self.vllm_config.model_config.max_model_len
            )
            if materialize_dcp_kv:
                max_prefill_buffer_size = self.vllm_config.model_config.max_model_len
            chunk_bounds = split_prefill_chunks(
                prefill_seq_lens_cpu, max_prefill_buffer_size
            )

            prefill_chunks = []
            for chunk_start, chunk_end in chunk_bounds:
                # Adjust workspace_starts in-place per chunk to be
                # 0-indexed within each chunk
                # Example: seq_lens=[10,15,20,5], chunks=[[0,2],[2,4]]
                #   Initial: workspace_starts=[0,10,25,45]
                #   After:   workspace_starts=[0,10,0,20]
                #           (chunk 0 starts at 0, chunk 1 starts at 0)
                offset = prefill_workspace_starts_cpu[chunk_start].item()
                prefill_workspace_starts_cpu[chunk_start:chunk_end] -= offset

                chunk_tot_seqlen = prefill_seq_lens_cpu[chunk_start:chunk_end].sum()
                token_start = query_start_loc_cpu[num_decodes + chunk_start].item()
                token_end = query_start_loc_cpu[num_decodes + chunk_end].item()
                tokens_slice = slice(token_start, token_end)

                # Create chunk view of gpu tensor
                chunk_workspace_starts = prefill_workspace_starts[chunk_start:chunk_end]
                chunk_block_table = common_attn_metadata.block_table_tensor[
                    num_decodes + chunk_start : num_decodes + chunk_end
                ]

                local_cu_seq_lens = None
                dcp_workspace_starts = None
                dcp_owner_stride = 0
                local_total_seq_len = 0
                if materialize_dcp_kv:
                    assert dcp_seq_lens_cpu is not None
                    chunk_dcp_lens = dcp_seq_lens_cpu[chunk_start:chunk_end]
                    dcp_starts_cpu = torch.zeros_like(chunk_dcp_lens)
                    if chunk_end - chunk_start > 1:
                        torch.cumsum(
                            chunk_dcp_lens[:-1],
                            dim=0,
                            out=dcp_starts_cpu[1:],
                        )
                    owner_totals = chunk_dcp_lens.sum(dim=0)
                    dcp_owner_stride = int(owner_totals.max().item())
                    local_total_seq_len = int(owner_totals[self.dcp_rank].item())
                    local_cu_cpu = torch.empty(
                        chunk_end - chunk_start + 1,
                        dtype=torch.int32,
                    )
                    local_cu_cpu[0] = 0
                    torch.cumsum(
                        chunk_dcp_lens[:, self.dcp_rank],
                        dim=0,
                        out=local_cu_cpu[1:],
                    )
                    local_cu_seq_lens = local_cu_cpu.to(self.device, non_blocking=True)
                    dcp_workspace_starts = dcp_starts_cpu.to(
                        self.device, non_blocking=True
                    )

                prefill_chunks.append(
                    FP8Meta.Prefill.Chunk(
                        tokens_slice=tokens_slice,
                        block_table=chunk_block_table,
                        req_start_idx=chunk_start,
                        workspace_starts=chunk_workspace_starts,
                        chunk_tot_seqlen=chunk_tot_seqlen,
                        local_cu_seq_lens=local_cu_seq_lens,
                        dcp_workspace_starts=dcp_workspace_starts,
                        dcp_owner_stride=dcp_owner_stride,
                        local_total_seq_len=local_total_seq_len,
                    )
                )

            prefill_workspace_starts.copy_(
                prefill_workspace_starts_cpu, non_blocking=True
            )

            fp8_metadata.prefill = FP8Meta.Prefill(
                request_ids=prefill_request_id,
                workspace_starts=prefill_workspace_starts,
                chunks=prefill_chunks,
            )

        if num_decodes > 0:
            # Use padded head count since that's what the kernel will see
            scheduler_metadata, _ = get_mla_metadata()

            kernel_meta = FlashMLASparseMetadata.FP8KernelMetadata(
                scheduler_metadata=scheduler_metadata,
                dummy_block_table=self.dummy_block_table[:active_num_decodes],
                cache_lens=self.max_model_len_tensor[:active_num_decodes],
            )
            fp8_metadata.decode = FP8Meta.Decode(
                seq_lens=common_attn_metadata.seq_lens[:active_num_decodes],
                kernel_metadata=kernel_meta,
                decode_query_len=decode_query_len,
            )

        return fp8_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        metadata.pcp_prefill_mode = self.pcp_prefill_mode
        if (
            self.pcp_dcp
            and self.pcp_prefill_mode == "kv_materialize"
            and metadata.num_prefills > 0
            and metadata.num_decodes > 0
        ):
            raise RuntimeError(
                "Sparse MLA kv_materialize does not support mixed "
                "prefill/decode batches."
            )
        if self.pcp_dcp:
            metadata.fp8_use_mixed_batch = (
                self.pcp_prefill_mode == "q_route" or metadata.num_prefills == 0
            )
        else:
            metadata.fp8_use_mixed_batch = self.fp8_default_mixed_batch
        if self.use_fp8_kv_cache:
            if metadata.fp8_use_mixed_batch:
                metadata.fp8_extra_metadata = self._build_fp8_mixed_decode_prefill(
                    common_attn_metadata
                )
            else:
                metadata.fp8_extra_metadata = self._build_fp8_separate_prefill_decode(
                    common_attn_metadata, metadata
                )

        return metadata


class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
    can_return_lse_for_decode: bool = True

    @staticmethod
    def _compute_fp8_decode_padded_heads(num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128
        # Compute padded head count for decode
        return 64 if num_heads <= 64 else 128

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        self.softmax_scale = scale
        # Prefill BF16 kernel requires 64 on Hopper, 128 on Blackwell
        self.prefill_padding = (
            128 if current_platform.is_device_capability_family(100) else 64
        )
        self.fp8_decode_padded_heads = self._compute_fp8_decode_padded_heads(num_heads)

        vllm_config = get_current_vllm_config()
        self.pcp_prefill_mode = vllm_config.attention_config.sparse_mla_pcp_prefill_mode
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        q_concat_shape = (max_tokens, num_heads, head_size)
        if is_quantized_kv_cache(kv_cache_dtype):
            assert kv_cache_dtype == "fp8_ds_mla", (
                "FlashMLA Sparse Attention backend fp8 only supports "
                "fp8_ds_mla kv-cache dtype"
            )

        if self.need_to_return_lse_for_decode and not is_quantized_kv_cache(
            kv_cache_dtype
        ):
            raise NotImplementedError(
                "DCP for FLASHMLA_SPARSE requires an fp8_ds_mla KV cache."
            )

        if kv_cache_dtype == "fp8_ds_mla":
            # Reserve workspace during initialization
            assert vllm_config is not None and vllm_config.model_config is not None
            prefill_workspace_size = get_prefill_workspace_size(
                vllm_config.model_config.max_model_len
            )
            self.prefill_workspace_shape = (prefill_workspace_size, head_size)
            workspace_specs = [
                (q_concat_shape, torch.bfloat16),
                (self.prefill_workspace_shape, torch.bfloat16),
            ]
            materialize_kv = (
                self.dcp_world_size > 1
                and self.pcp_world_size > 1
                and self.pcp_prefill_mode == "kv_materialize"
            )
            if materialize_kv:
                fp8_workspace_rows = (
                    self.dcp_world_size + 1
                ) * vllm_config.model_config.max_model_len
                workspace_specs.append(
                    (
                        (fp8_workspace_rows, FP8_DS_MLA_ENTRY_BYTES),
                        torch.uint8,
                    )
                )
            workspaces = current_workspace_manager().get_simultaneous(*workspace_specs)
            self.q_concat_buffer = workspaces[0]
            self.prefill_bf16_workspace = workspaces[1]
            self.prefill_fp8_workspace = workspaces[2] if materialize_kv else None
        else:
            (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
                (q_concat_shape, torch.bfloat16),
            )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill). req_id_per_token covers the whole batch; slice it
        # to the MQA tokens (q may exclude prefill tokens routed to dense MHA).
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: topk_indices.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        return self._bf16_flash_mla_kernel(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            topk_length,
        )

    def _forward_fp8_kv_separate_prefill_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)
        materialize_dcp_kv = (
            self.dcp_world_size > 1
            and self.pcp_world_size > 1
            and attn_metadata.pcp_prefill_mode == "kv_materialize"
            and fp8_metadata.num_prefills > 0
        )
        if materialize_dcp_kv:
            return self._forward_fp8_kv_materialized_prefill(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                attn_metadata,
                fp8_metadata,
            )

        num_decodes = fp8_metadata.num_decodes
        num_mqa_tokens = q.shape[0]
        num_decode_tokens = fp8_metadata.num_decode_tokens
        num_prefill_tokens = num_mqa_tokens - num_decode_tokens
        assert num_prefill_tokens in (0, fp8_metadata.num_prefill_tokens), (
            "FP8 sparse MLA expects either the decode subset or the full batch"
        )

        prefill_request_ids = None
        prefill_workspace_starts = None
        has_prefill_workspace = False
        if num_prefill_tokens > 0:
            assert fp8_metadata.prefill is not None
            prefill_request_ids = fp8_metadata.prefill.request_ids
            prefill_workspace_starts = fp8_metadata.prefill.workspace_starts
            has_prefill_workspace = True

        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        # For FP8 cache: prefill uses workspace mapping (upconverted to BF16)
        # For BF16 cache: always use global cache slots (no workspace)
        # prefill_workspace_starts has been adjusted in-place per chunk so
        # prefill indices automatically come out chunk-local
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: topk_indices.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            HAS_PREFILL_WORKSPACE=has_prefill_workspace,
            prefill_workspace_request_ids=prefill_request_ids,
            prefill_workspace_starts=prefill_workspace_starts,
            return_valid_counts=True,
        )

        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)

        def _fp8_decode(
            q: torch.Tensor,
            topk_indices: torch.Tensor,
        ) -> torch.Tensor:
            # Reshape q: (num_decode_tokens, num_heads, head_dim)
            #         -> (num_decodes, seq_len, num_heads, head_dim)
            q = reshape_query_for_spec_decode(q, num_decodes)
            seq_len = q.shape[1]
            # Reshape topk_indices: (num_decode_tokens, topk)
            #                    -> (num_decodes, seq_len, topk)
            topk_indices = topk_indices.view(num_decodes, seq_len, -1)
            assert fp8_metadata.decode is not None
            attn_out, _ = self._fp8_flash_mla_kernel(
                q=q,
                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
                topk_indices=topk_indices,
                kernel_metadata=fp8_metadata.decode.kernel_metadata,
            )
            # Reshape output: (num_decodes, seq_len, num_heads, head_dim_v)
            #              -> (num_decode_tokens, num_heads, head_dim_v)
            return reshape_attn_output_for_spec_decode(attn_out)

        # Pure decode: direct call without allocation
        if num_decode_tokens > 0 and num_prefill_tokens == 0:
            assert fp8_metadata.decode is not None
            attn_out = _fp8_decode(q, topk_indices)
        else:
            # Mixed or pure prefill: allocate output tensor
            attn_out = q.new_empty(
                (num_mqa_tokens, self.num_heads, self.kv_lora_rank),
                dtype=q.dtype,
                device=q.device,
            )

            if num_decode_tokens > 0:
                attn_out[:num_decode_tokens] = _fp8_decode(
                    q[:num_decode_tokens],
                    topk_indices[:num_decode_tokens],
                )

            assert fp8_metadata.prefill is not None
            for chunk in fp8_metadata.prefill.chunks:
                chunk_workspace = self.prefill_bf16_workspace[: chunk.chunk_tot_seqlen]
                ops.cp_gather_and_upconvert_fp8_kv_cache(
                    kv_c_and_k_pe_cache,
                    chunk_workspace,
                    chunk.block_table,
                    chunk.workspace_starts,
                    len(chunk.block_table),
                )

                chunk_q = q[chunk.tokens_slice]
                chunk_topk_indices_workspace = topk_indices[chunk.tokens_slice]
                chunk_topk_length = topk_length[chunk.tokens_slice]

                attn_out[chunk.tokens_slice] = self._bf16_flash_mla_kernel(
                    chunk_q,
                    chunk_workspace,
                    chunk_topk_indices_workspace,
                    chunk_topk_length,
                )

        return attn_out

    def _forward_fp8_kv_materialized_prefill(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        fp8_metadata: FlashMLASparseMetadata.FP8SeparatePrefillDecode,
    ) -> torch.Tensor:
        if fp8_metadata.num_decodes > 0:
            raise RuntimeError(
                "Sparse MLA kv_materialize does not support mixed "
                "prefill/decode batches."
            )
        routing = attn_metadata.pcp_query_routing
        if routing is None:
            raise RuntimeError(
                "Sparse MLA kv_materialize requires global request metadata."
            )
        if self.prefill_fp8_workspace is None:
            raise RuntimeError("Sparse MLA kv_materialize workspace is unavailable.")
        assert fp8_metadata.prefill is not None

        attn_out = q.new_empty(
            (q.shape[0], self.num_heads, self.kv_lora_rank),
        )
        dcp_group = get_dcp_group()

        for chunk in fp8_metadata.prefill.chunks:
            assert chunk.local_cu_seq_lens is not None
            assert chunk.dcp_workspace_starts is not None
            owner_stride = chunk.dcp_owner_stride
            raw_rows = (self.dcp_world_size + 1) * owner_stride
            if raw_rows > self.prefill_fp8_workspace.shape[0]:
                raise RuntimeError(
                    "Sparse MLA kv_materialize FP8 workspace is too small."
                )
            if (
                self.dcp_world_size * owner_stride
                > self.prefill_bf16_workspace.shape[0]
            ):
                raise RuntimeError(
                    "Sparse MLA kv_materialize BF16 workspace is too small."
                )

            local_raw = self.prefill_fp8_workspace[:owner_stride]
            gathered_raw = self.prefill_fp8_workspace[owner_stride:raw_rows]
            local_raw.zero_()
            if chunk.local_total_seq_len > 0:
                with record_function_or_nullcontext(
                    "flashmla_sparse.compute.kv_materialize.main_kv_pack"
                ):
                    ops.cp_gather_cache(
                        kv_c_and_k_pe_cache,
                        local_raw[: chunk.local_total_seq_len],
                        chunk.block_table,
                        chunk.local_cu_seq_lens,
                        len(chunk.block_table),
                    )
            with record_function_or_nullcontext(
                "flashmla_sparse.comm.kv_materialize.main_kv_allgather"
            ):
                torch.distributed.all_gather_into_tensor(
                    gathered_raw,
                    local_raw,
                    group=dcp_group.device_group,
                )

            gathered_bf16 = self.prefill_bf16_workspace[
                : self.dcp_world_size * owner_stride
            ]
            with record_function_or_nullcontext(
                "flashmla_sparse.compute.kv_materialize.main_kv_upconvert"
            ):
                upconvert_fp8_ds_mla_workspace(gathered_raw, gathered_bf16)

            local_rows, chunk_rows = get_pcp_local_rows_for_range(
                routing,
                chunk.tokens_slice.start,
                chunk.tokens_slice.stop,
            )
            if local_rows.shape[0] == 0:
                continue
            chunk_topk = topk_indices.index_select(0, local_rows)
            request_ids = fp8_metadata.prefill.request_ids.index_select(
                0, chunk_rows + chunk.tokens_slice.start
            )
            workspace_topk, topk_length = map_materialized_dcp_topk(
                chunk_topk,
                request_ids,
                chunk.dcp_workspace_starts,
                owner_stride,
                self.dcp_world_size,
                attn_metadata.cp_kv_cache_interleave_size,
                request_offset=chunk.req_start_idx,
            )
            chunk_output = self._bf16_flash_mla_kernel(
                q.index_select(0, local_rows),
                gathered_bf16,
                workspace_topk,
                topk_length,
            )
            attn_out.index_copy_(0, local_rows, chunk_output)

        return attn_out

    def _forward_fp8_kv_mixed_batch(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mixed batch FP8 forward path that treats all tokens as one batch.

        This is equivalent to main branch's approach and avoids the BF16
        prefill kernel which has head padding overhead when num_heads is small.
        Used when use_mixed_batch is True.
        """
        if self.dcp_world_size > 1:
            topk_indices = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[: topk_indices.shape[0]],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                compact_valid_to_front=False,
            )
        else:
            topk_indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[: topk_indices.shape[0]],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )

        assert attn_metadata.fp8_extra_metadata is not None
        assert isinstance(
            attn_metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
        )
        fp8_metadata = attn_metadata.fp8_extra_metadata

        _attn_out, _lse = self._fp8_flash_mla_kernel(
            q=q.unsqueeze(0),
            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
            topk_indices=topk_indices.unsqueeze(0),
            kernel_metadata=fp8_metadata,
        )

        out = _attn_out.squeeze(0)
        if not self.need_to_return_lse_for_decode:
            return out, None

        assert _lse is not None
        lse = _lse.squeeze(0).transpose(0, 1)
        empty_rows = (topk_indices == -1).all(dim=-1)
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out.contiguous(), lse

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self.fp8_decode_padded_heads

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
        )

        # Slice output and LSE back to actual head count if we padded.
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]
            lse = lse[:, :actual_num_heads, :]

        return out, lse

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        if self.num_heads % self.prefill_padding != 0:
            assert self.prefill_padding % self.num_heads == 0
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to "
                f"{self.prefill_padding} for BF16 sparse prefill kernel"
            )
            q_padded = q.new_empty((q.shape[0], self.prefill_padding, q.shape[2]))
            q_padded[:, : self.num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            topk_length=topk_length,
        )[0]

        output = output[:, : self.num_heads, :]
        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        pcp_prefill_mode = getattr(attn_metadata, "pcp_prefill_mode", "q_route")
        if (
            self.dcp_world_size > 1
            and self.pcp_world_size > 1
            and attn_metadata.num_prefills > 0
            and attn_metadata.pcp_query_routing is None
        ):
            raise RuntimeError(
                f"FLASHMLA_SPARSE {pcp_prefill_mode} requires global request metadata."
            )
        if (
            pcp_prefill_mode == "kv_materialize"
            and attn_metadata.num_prefills > 0
            and attn_metadata.fp8_use_mixed_batch
        ):
            raise RuntimeError(
                "Sparse MLA kv_materialize requires the separate FP8 prefill path."
            )

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]

        # Get topk indices
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"

        lse: torch.Tensor | None = None
        if not use_fp8_cache:
            attn_out = self._forward_bf16_kv(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        elif attn_metadata.fp8_use_mixed_batch:
            attn_out, lse = self._forward_fp8_kv_mixed_batch(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        else:
            attn_out = self._forward_fp8_kv_separate_prefill_decode(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )

        return attn_out, lse
