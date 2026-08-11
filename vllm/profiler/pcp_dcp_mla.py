# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary profiling hooks for the PCP+DCP MLA prefix-prefill experiment."""

import importlib.metadata
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any
from urllib.parse import quote

import regex as re
import torch
import torch.distributed as dist

_ENV_VAR = "PCP_DCP_MLA_PROFILE"
_LABEL_PREFIX = "VLLM_PCP_DCP_MLA"
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class _PendingRecord:
    category: str
    fields: dict[str, Any]
    message: str
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None


@dataclass
class _ProfileState:
    active_label: str | None = None
    records: list[_PendingRecord] = field(default_factory=list)


_STATE = _ProfileState()


def is_enabled() -> bool:
    return os.getenv(_ENV_VAR, "0") == "1"


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _json_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, torch.dtype | torch.device):
        return str(value)
    if isinstance(value, torch.Size | tuple | list):
        return "x".join(str(item) for item in value)
    return str(value)


def _message(category: str, fields: dict[str, Any]) -> str:
    values = {
        "category": category,
        "iteration": _STATE.active_label,
        "rank": _rank(),
        **fields,
    }
    encoded = [
        f"{key}={quote(str(_json_value(value)), safe='._,:x-')}"
        for key, value in values.items()
    ]
    return "|".join((_LABEL_PREFIX, *encoded))


def arm(label: str) -> dict[str, Any]:
    if not is_enabled():
        raise RuntimeError(f"Set {_ENV_VAR}=1 before importing vLLM.")
    if not _SAFE_LABEL.fullmatch(label):
        raise ValueError(f"Invalid experiment label: {label!r}")
    if _STATE.active_label is not None:
        raise RuntimeError(
            f"Profiling label {_STATE.active_label!r} has not been collected."
        )
    _STATE.active_label = label
    _STATE.records.clear()
    return {"rank": _rank(), "label": label, "armed": True}


@contextmanager
def timed_range(category: str, **fields: Any):
    if _STATE.active_label is None:
        yield
        return

    normalized = {key: _json_value(value) for key, value in fields.items()}
    message = _message(category, normalized)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(message)
    start_event.record()
    try:
        yield
    finally:
        end_event.record()
        torch.cuda.nvtx.range_pop()
        _STATE.records.append(
            _PendingRecord(
                category=category,
                fields=normalized,
                message=message,
                start_event=start_event,
                end_event=end_event,
            )
        )


@contextmanager
def annotation_range(category: str, **fields: Any):
    """Emit an NVTX range without inserting CUDA timing events."""
    if _STATE.active_label is None:
        yield
        return

    normalized = {key: _json_value(value) for key, value in fields.items()}
    message = _message(category, normalized)
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
        _STATE.records.append(
            _PendingRecord(category=category, fields=normalized, message=message)
        )


def mark(category: str, **fields: Any) -> None:
    if _STATE.active_label is None:
        return
    normalized = {key: _json_value(value) for key, value in fields.items()}
    message = _message(category, normalized)
    torch.cuda.nvtx.mark(message)
    _STATE.records.append(
        _PendingRecord(category=category, fields=normalized, message=message)
    )


def collect(*, disarm: bool = True) -> dict[str, Any]:
    label = _STATE.active_label
    if label is None:
        raise RuntimeError("No experiment iteration is armed.")
    torch.accelerator.synchronize()
    records = []
    for record in _STATE.records:
        elapsed_ms = None
        if record.start_event is not None and record.end_event is not None:
            elapsed_ms = record.start_event.elapsed_time(record.end_event)
        records.append(
            {
                "category": record.category,
                "fields": record.fields,
                "message": record.message,
                "elapsed_ms": elapsed_ms,
            }
        )
    result = {
        "rank": _rank(),
        "device": torch.accelerator.current_device_index(),
        "label": label,
        "records": records,
    }
    if disarm:
        _STATE.active_label = None
        _STATE.records.clear()
    return result


def _worker_metadata(worker) -> dict[str, Any]:
    import vllm
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer

    config = worker.vllm_config
    model = worker.get_model()
    mla_layers = [
        module for module in model.modules() if isinstance(module, MLAAttention)
    ]
    decoder_layers = [
        module
        for module in model.modules()
        if isinstance(module, DeepseekV2DecoderLayer)
    ]
    backend_rows = []
    for layer in mla_layers:
        prefill_backend = layer.prefill_backend
        backend_rows.append(
            {
                "layer_name": layer.layer_name,
                "outer_backend": layer.attn_backend.get_name(),
                "outer_impl": type(layer.impl).__name__,
                "prefill_backend": (
                    prefill_backend.get_name() if prefill_backend is not None else None
                ),
                "prefill_backend_class": (
                    type(prefill_backend).__name__
                    if prefill_backend is not None
                    else None
                ),
                "kv_cache_dtype": layer.kv_cache_dtype,
                "use_pcp": layer.use_pcp,
                "use_sparse": layer.use_sparse,
            }
        )
    named_parameters = list(model.named_parameters())
    parameter_dtypes = sorted(
        {str(parameter.dtype) for _, parameter in named_parameters}
    )
    parameter_dtype_numel: dict[str, int] = {}
    for _, parameter in named_parameters:
        dtype = str(parameter.dtype)
        parameter_dtype_numel[dtype] = (
            parameter_dtype_numel.get(dtype, 0) + parameter.numel()
        )
    non_bf16_parameters = [
        {
            "name": name,
            "dtype": str(parameter.dtype),
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
        }
        for name, parameter in named_parameters
        if parameter.dtype != torch.bfloat16
    ]
    nccl_version = torch.cuda.nccl.version() if torch.cuda.is_available() else None
    return {
        "rank": _rank(),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "device": torch.accelerator.current_device_index(),
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        "nccl_version": list(nccl_version) if nccl_version is not None else None,
        "vllm_version": str(vllm.__version__),
        "vllm_distribution_version": importlib.metadata.version("vllm"),
        "parameter_dtypes": parameter_dtypes,
        "parameter_dtype_numel": parameter_dtype_numel,
        "non_bf16_parameters": non_bf16_parameters,
        "load_format": str(config.load_config.load_format),
        "model_dtype": str(config.model_config.dtype),
        "quantization": config.model_config.quantization,
        "parallel": {
            "tp": config.parallel_config.tensor_parallel_size,
            "pcp": config.parallel_config.prefill_context_parallel_size,
            "dcp": config.parallel_config.decode_context_parallel_size,
            "dcp_comm_backend": config.parallel_config.dcp_comm_backend,
            "cp_kv_cache_interleave_size": (
                config.parallel_config.cp_kv_cache_interleave_size
            ),
        },
        "cache": {
            "block_size": config.cache_config.block_size,
            "kv_cache_dtype": config.cache_config.cache_dtype,
            "enable_prefix_caching": config.cache_config.enable_prefix_caching,
        },
        "scheduler": {
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "max_model_len": config.model_config.max_model_len,
        },
        "compilation": {
            "mode": str(config.compilation_config.mode),
            "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
            "enforce_eager": config.model_config.enforce_eager,
        },
        "backends": backend_rows,
        "decoder_layers": [
            {
                "layer_idx": layer.layer_idx,
                "attention_type": type(layer.self_attn).__name__,
                "mlp_type": type(layer.mlp).__name__,
            }
            for layer in decoder_layers
        ],
    }


class PCPDCPMLAWorkerExtension:
    """Trusted string-addressed RPCs used by the local experiment driver."""

    def pcp_dcp_mla_profile_arm(self, label: str) -> dict[str, Any]:
        return arm(label)

    def pcp_dcp_mla_profile_collect(self) -> dict[str, Any]:
        return collect()

    def pcp_dcp_mla_profile_metadata(self) -> dict[str, Any]:
        return _worker_metadata(self)


def profile_scope(scope: str):
    def decorate(func):
        if not is_enabled():
            return func

        @wraps(func)
        def wrapped(module, *args, **kwargs):
            positions = args[0] if args else kwargs.get("positions")
            hidden_states = args[1] if len(args) > 1 else kwargs.get("hidden_states")
            layer = getattr(module, "layer_idx", None)
            if layer is None:
                mla_wrapper = getattr(module, "mla_attn", None)
                layer = getattr(mla_wrapper, "prefix", "unknown")
            fields = {
                "scope": scope,
                "layer": layer,
                "local_q_tokens": positions.numel()
                if isinstance(positions, torch.Tensor)
                else "unknown",
                "activation_dtype": hidden_states.dtype
                if isinstance(hidden_states, torch.Tensor)
                else "unknown",
            }
            with timed_range("scope", **fields):
                return func(module, *args, **kwargs)

        return wrapped

    return decorate
