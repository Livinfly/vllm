# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the dense MLA PCP+DCP cached-prefix prefill experiment."""

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PCP_DCP_MLA_PROFILE", "1")
os.environ.setdefault("VLLM_NVTX_SCOPES_FOR_PROFILING", "1")

CONTRACT_PREFIX_TOKENS = 81_920
SMOKE_PREFIX_TOKENS = 1_024
SUFFIX_TOKENS = 256
MAX_MODEL_LEN = 131_072
PCP_SIZE = 2
DCP_SIZE = 2
BLOCK_SIZE = 64
HIDDEN_SIZE = 7_168
NUM_ATTENTION_HEADS = 128
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
EXPERIMENT_ROOT = Path("benchmarks/experimental/pcp_dcp_mla_prefix")
EXPECTED_GIT_SHA = "c810e5ee9976ad86b81d1277b53e76d0ee639414"
REQUIRED_PCP_COMMIT = "b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd"


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return {"command": command, "returncode": 127, "stderr": str(exc)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_output(*args: str) -> str | None:
    result = _run_command(["git", *args])
    if result["returncode"] != 0:
        return None
    return result.get("stdout", "")


def _validate_repository(expected_sha: str) -> dict[str, Any]:
    head = _git_output("rev-parse", "HEAD")
    source_ancestor = _run_command(
        ["git", "merge-base", "--is-ancestor", expected_sha, "HEAD"]
    )
    if source_ancestor["returncode"] != 0:
        raise RuntimeError(
            f"Precompiled source SHA {expected_sha} is not an ancestor of {head}."
        )
    ancestor = _run_command(
        ["git", "merge-base", "--is-ancestor", REQUIRED_PCP_COMMIT, "HEAD"]
    )
    if ancestor["returncode"] != 0:
        raise RuntimeError(
            f"HEAD does not contain required PCP commit {REQUIRED_PCP_COMMIT} "
            "from PR #46570."
        )
    return {
        "experiment_git_sha": head,
        "precompiled_source_sha": expected_sha,
        "precompiled_source_is_ancestor": True,
        "required_pr": "https://github.com/vllm-project/vllm/pull/46570",
        "required_commit": REQUIRED_PCP_COMMIT,
        "required_commit_is_ancestor": True,
    }


def _snapshot_patch(output_path: Path) -> None:
    parts = []
    for args in (("diff", "--binary"), ("diff", "--cached", "--binary")):
        diff = _git_output(*args)
        if diff:
            parts.append(diff)

    untracked = _git_output("ls-files", "--others", "--exclude-standard") or ""
    included_roots = (
        str(EXPERIMENT_ROOT) + "/",
        "vllm/profiler/pcp_dcp_mla.py",
    )
    for path in sorted(untracked.splitlines()):
        if not path.startswith(included_roots):
            continue
        result = _run_command(
            ["git", "diff", "--no-index", "--binary", "/dev/null", path]
        )
        if result.get("stdout"):
            parts.append(result["stdout"])
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _token_hash(token_ids: list[int]) -> str:
    values = array("I", token_ids)
    if sys.byteorder != "little":
        values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _make_prefix(length: int, vocab_size: int, seed: int) -> list[int]:
    usable_vocab = vocab_size - 512
    if usable_vocab <= 0:
        raise ValueError(f"Vocab size is too small: {vocab_size}")
    return [512 + ((seed + index * 48_271) % usable_vocab) for index in range(length)]


def _make_suffix(
    length: int,
    vocab_size: int,
    seed: int,
    variant: int,
) -> list[int]:
    if length <= 0:
        raise ValueError("Suffix length must be positive.")
    rng = random.Random(seed + variant * 1_000_003)
    suffix = [512 + ((variant * 97) % (vocab_size - 512))]
    suffix.extend(rng.randrange(512, vocab_size) for _ in range(length - 1))
    return suffix


def _collect_environment() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "resolved_command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_NVTX_SCOPES_FOR_PROFILING",
                "PCP_DCP_MLA_PROFILE",
                "NCCL_DEBUG",
                "NCCL_ALGO",
                "NCCL_PROTO",
            )
        },
        "git": {
            "head": _git_output("rev-parse", "HEAD"),
            "origin_main": _git_output("rev-parse", "origin/main"),
            "status": _git_output("status", "--short", "--branch"),
            "commit": _git_output("show", "-s", "--format=fuller", "HEAD"),
        },
        "commands": {
            "nvidia_smi": _run_command(["nvidia-smi"]),
            "nvidia_smi_topology": _run_command(["nvidia-smi", "topo", "-m"]),
            "nvidia_smi_clocks_power": _run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,pci.bus_id,driver_version,pstate,"
                    "clocks.sm,clocks.mem,power.draw,power.limit,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "nvcc": _run_command(["nvcc", "--version"]),
            "nsys": _run_command(["nsys", "--version"]),
        },
    }


def _validate_hf_config(config: dict[str, Any]) -> None:
    expected = {
        "num_hidden_layers": 1,
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"DeepSeek-V3 geometry mismatch: {mismatches}")
    if "index_topk" in config:
        raise RuntimeError("The selected config contains index_topk and is sparse MLA.")
    if config.get("first_k_dense_replace", 0) <= 0:
        raise RuntimeError("Layer 0 is not configured as a dense MLP layer.")
    if config.get("quantization_config") is not None:
        raise RuntimeError("quantization_config must be None for dummy BF16 weights.")


def _validate_worker_metadata(
    rows: list[dict[str, Any]],
    expect_outer_backend: str,
    expected_source_sha: str,
) -> None:
    if {row["rank"] for row in rows} != {0, 1}:
        raise RuntimeError(f"Expected ranks 0 and 1, got: {rows}")
    for row in rows:
        if row["world_size"] != 2:
            raise RuntimeError(f"Expected world size 2: {row}")
        if row["parameter_dtypes"] != ["torch.bfloat16"]:
            raise RuntimeError(f"Weights are not exclusively BF16: {row}")
        distribution_version = row["vllm_distribution_version"]
        wheel_commit = (
            distribution_version.split("+g", 1)[1].split(".", 1)[0]
            if "+g" in distribution_version
            else ""
        )
        if len(wheel_commit) < 7 or not expected_source_sha.startswith(wheel_commit):
            raise RuntimeError(
                "Installed vLLM wheel does not match the frozen source commit: "
                f"expected {expected_source_sha}, got {distribution_version}"
            )
        if row["quantization"] is not None:
            raise RuntimeError(f"Unexpected quantization: {row}")
        if row["decoder_layers"] != [
            {
                "layer_idx": 0,
                "attention_type": "DeepseekV2MLAAttention",
                "mlp_type": "DeepseekV2MLP",
            }
        ]:
            raise RuntimeError(f"Expected one layer-0-shaped dense layer: {row}")
        if len(row["backends"]) != 1:
            raise RuntimeError(f"Expected exactly one MLA attention layer: {row}")
        backend = row["backends"][0]
        if backend["use_sparse"]:
            raise RuntimeError(f"Sparse MLA was selected: {backend}")
        if expect_outer_backend != "ANY" and (
            backend["outer_backend"] != expect_outer_backend
        ):
            raise RuntimeError(
                "Automatic backend selection did not meet the experiment contract: "
                f"expected {expect_outer_backend}, got {backend['outer_backend']}. "
                "Do not force a replacement silently; record and resolve the mismatch."
            )


def _record_sum(profile: dict[str, Any], category: str, field: str) -> int:
    return sum(
        int(record["fields"].get(field, 0))
        for record in profile["records"]
        if record["category"] == category
    )


def _expected_context_comm(
    rank: int,
    prefix_tokens: int,
    suffix_tokens: int,
) -> dict[str, Any]:
    num_chunks = 2 * PCP_SIZE
    chunk_size = math.ceil(suffix_tokens / num_chunks)
    chunk_indices = (rank, num_chunks - 1 - rank)
    context_tokens = [
        prefix_tokens + chunk_index * chunk_size
        for chunk_index in chunk_indices
        if chunk_index * chunk_size < suffix_tokens
    ]
    if any(tokens % DCP_SIZE != 0 for tokens in context_tokens):
        raise RuntimeError(
            "The contract shape must divide every virtual context across DCP."
        )
    local_context_tokens = sum(tokens // DCP_SIZE for tokens in context_tokens)
    bytes_per_token = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2
    unique_prefix_send_bytes = prefix_tokens // DCP_SIZE * bytes_per_token
    physical_send_bytes = local_context_tokens * bytes_per_token
    return {
        "pcp_chunk_indices": list(chunk_indices),
        "virtual_context_tokens": context_tokens,
        "physical_global_context_rows": sum(context_tokens),
        "physical_local_context_rows": local_context_tokens,
        "physical_send_bytes": physical_send_bytes,
        "unique_prefix_send_bytes": unique_prefix_send_bytes,
        "physical_to_unique_prefix_ratio": (
            physical_send_bytes / unique_prefix_send_bytes
        ),
    }


def _validate_iteration_profile(
    profiles: list[dict[str, Any]],
    prefix_tokens: int,
    suffix_tokens: int,
) -> list[dict[str, Any]]:
    expected_local_q = math.ceil(suffix_tokens / (2 * PCP_SIZE)) * 2
    expected_suffix_bytes = (
        suffix_tokens // PCP_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2
    )
    expected_hidden_bytes = suffix_tokens // PCP_SIZE * HIDDEN_SIZE * 2
    if {profile["rank"] for profile in profiles} != {0, 1}:
        raise RuntimeError(f"Missing rank profile: {profiles}")
    validation = []
    for profile in profiles:
        expected_context = _expected_context_comm(
            int(profile["rank"]), prefix_tokens, suffix_tokens
        )
        scope_records = [
            record for record in profile["records"] if record["category"] == "scope"
        ]
        scopes = [record["fields"].get("scope") for record in scope_records]
        if sorted(scopes) != ["full_layer", "self_attn"]:
            raise RuntimeError(f"Unexpected scope records: {scope_records}")
        if any(
            int(record["fields"]["local_q_tokens"]) != expected_local_q
            for record in scope_records
        ):
            raise RuntimeError(f"PCP local Q shape mismatch: {scope_records}")
        partition_records = [
            record
            for record in profile["records"]
            if record["category"] == "pcp_partition"
        ]
        if len(partition_records) != 1:
            raise RuntimeError(f"Expected one scheduler step: {partition_records}")
        actual_context_bytes = _record_sum(
            profile, "context_attention_comm", "send_bytes"
        )
        if actual_context_bytes != expected_context["physical_send_bytes"]:
            raise RuntimeError(f"DCP prefix payload mismatch: {profile}")
        actual_suffix_bytes = _record_sum(profile, "suffix_cache_comm", "send_bytes")
        if actual_suffix_bytes != expected_suffix_bytes:
            raise RuntimeError(f"PCP suffix payload mismatch: {profile}")
        actual_hidden_bytes = _record_sum(profile, "hidden_restore_comm", "send_bytes")
        if actual_hidden_bytes != expected_hidden_bytes:
            raise RuntimeError(f"PCP hidden-restore payload mismatch: {profile}")
        validation.append(
            {
                "rank": profile["rank"],
                "local_q_tokens": expected_local_q,
                "context_attention_comm": {
                    **expected_context,
                    "actual_send_bytes": actual_context_bytes,
                    "actual_recv_bytes": _record_sum(
                        profile, "context_attention_comm", "recv_bytes"
                    ),
                    "collective_count": sum(
                        record["category"] == "context_attention_comm"
                        for record in profile["records"]
                    ),
                },
                "suffix_cache_comm": {
                    "expected_send_bytes": expected_suffix_bytes,
                    "actual_send_bytes": actual_suffix_bytes,
                    "actual_recv_bytes": _record_sum(
                        profile, "suffix_cache_comm", "recv_bytes"
                    ),
                    "collective_count": sum(
                        record["category"] == "suffix_cache_comm"
                        for record in profile["records"]
                    ),
                },
                "hidden_restore_comm": {
                    "expected_send_bytes": expected_hidden_bytes,
                    "actual_send_bytes": actual_hidden_bytes,
                    "actual_recv_bytes": _record_sum(
                        profile, "hidden_restore_comm", "recv_bytes"
                    ),
                    "collective_count": sum(
                        record["category"] == "hidden_restore_comm"
                        for record in profile["records"]
                    ),
                },
            }
        )
    return validation


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _event_summary(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scope in ("self_attn", "full_layer"):
        max_rank_times = []
        per_iteration = []
        for iteration in iterations:
            rank_times = {}
            for profile in iteration["profiles"]:
                matching = [
                    record
                    for record in profile["records"]
                    if record["category"] == "scope"
                    and record["fields"].get("scope") == scope
                ]
                if len(matching) != 1:
                    raise RuntimeError(f"Expected one {scope} record: {matching}")
                rank_times[str(profile["rank"])] = matching[0]["elapsed_ms"]
            step_time = max(rank_times.values())
            max_rank_times.append(step_time)
            per_iteration.append(
                {
                    "label": iteration["label"],
                    "rank_ms": rank_times,
                    "max_rank_ms": step_time,
                }
            )
        summary[scope] = {
            "iterations": per_iteration,
            "max_rank_mean_ms": statistics.fmean(max_rank_times),
            "max_rank_p50_ms": _percentile(max_rank_times, 0.50),
            "max_rank_p90_ms": _percentile(max_rank_times, 0.90),
            "max_rank_stdev_ms": statistics.stdev(max_rank_times)
            if len(max_rank_times) > 1
            else 0.0,
        }
    return summary


def _request_evidence(
    output, prompt_ids: list[int], expected_cached: int
) -> dict[str, Any]:
    if output.num_cached_tokens != expected_cached:
        raise RuntimeError(
            f"Expected exactly {expected_cached} cached tokens, "
            f"got {output.num_cached_tokens}."
        )
    returned_prompt = output.prompt_token_ids
    if returned_prompt is None or returned_prompt != prompt_ids:
        raise RuntimeError("vLLM did not return the exact submitted prompt token IDs.")
    generated_tokens = list(output.outputs[0].token_ids)
    if len(generated_tokens) != 1:
        raise RuntimeError(f"Expected exactly one sampled token: {generated_tokens}")
    return {
        "request_id": output.request_id,
        "prompt_tokens": len(prompt_ids),
        "prompt_token_sha256_le_u32": _token_hash(prompt_ids),
        "num_cached_tokens": output.num_cached_tokens,
        "num_computed_prompt_tokens": len(prompt_ids) - output.num_cached_tokens,
        "num_cache_creation_tokens": output.num_cache_creation_tokens,
        "sampled_token_id": generated_tokens[0],
        "finished": output.finished,
    }


def _run_measured_request(
    llm,
    prompt,
    sampling_params,
    label: str,
    expected_cached: int,
    capture_cuda_profiler: bool,
) -> dict[str, Any]:
    armed = llm.collective_rpc("pcp_dcp_mla_profile_arm", args=(label,))
    if capture_cuda_profiler:
        llm.start_profile(label)
    start = time.perf_counter()
    try:
        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
    finally:
        if capture_cuda_profiler:
            llm.stop_profile()
    wall_ms = (time.perf_counter() - start) * 1_000
    profiles = llm.collective_rpc("pcp_dcp_mla_profile_collect")
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one output, got {len(outputs)}")
    return {
        "label": label,
        "armed": armed,
        "frontend_wall_ms": wall_ms,
        "request": _request_evidence(
            outputs[0], prompt["prompt_token_ids"], expected_cached
        ),
        "profiles": profiles,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "smoke-nsys", "events", "nsys"),
        default="events",
    )
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--revision")
    parser.add_argument("--hf-config-path")
    parser.add_argument("--prefix-tokens", type=int)
    parser.add_argument("--suffix-tokens", type=int, default=SUFFIX_TOKENS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--expect-outer-backend", default="FLASH_ATTN_MLA")
    parser.add_argument("--allow-non-contract-shape", action="store_true")
    parser.add_argument("--expected-git-sha", default=EXPECTED_GIT_SHA)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository_evidence = _validate_repository(args.expected_git_sha)
    smoke = args.mode.startswith("smoke")
    capture = args.mode in ("smoke-nsys", "nsys")
    prefix_tokens = args.prefix_tokens or (
        SMOKE_PREFIX_TOKENS if smoke else CONTRACT_PREFIX_TOKENS
    )
    if not args.allow_non_contract_shape:
        expected_prefix = SMOKE_PREFIX_TOKENS if smoke else CONTRACT_PREFIX_TOKENS
        if prefix_tokens != expected_prefix or args.suffix_tokens != SUFFIX_TOKENS:
            raise ValueError(
                f"Mode {args.mode} requires prefix={expected_prefix}, "
                f"suffix={SUFFIX_TOKENS}."
            )
    if prefix_tokens + args.suffix_tokens + 1 > MAX_MODEL_LEN:
        raise ValueError("Prompt plus sampled token exceeds max_model_len.")
    if args.suffix_tokens % PCP_SIZE != 0 or prefix_tokens % DCP_SIZE != 0:
        raise ValueError("Prefix/suffix lengths must divide evenly across CP ranks.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path(
        f"artifacts/pcp_dcp_mla_prefix/{timestamp}-{args.mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = _collect_environment()
    metadata["experiment"] = {
        "mode": args.mode,
        "model": args.model,
        "revision": args.revision,
        "hf_config_path": args.hf_config_path,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "max_model_len": MAX_MODEL_LEN,
        "weight_format": "dummy unquantized BF16",
        "activation_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "attention_backend_request": "auto",
        "mla_prefill_backend_request": "auto",
        "expected_outer_backend": args.expect_outer_backend,
        "repository_requirement": repository_evidence,
    }
    _write_json(output_dir / "environment.json", metadata)
    _snapshot_patch(output_dir / "local_changes.patch")

    engine_kwargs = {
        "model": args.model,
        "revision": args.revision,
        "skip_tokenizer_init": True,
        "load_format": "dummy",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "hf_overrides": {
            "num_hidden_layers": 1,
            "num_nextn_predict_layers": 0,
            "quantization_config": None,
        },
        "tensor_parallel_size": 1,
        "prefill_context_parallel_size": PCP_SIZE,
        "decode_context_parallel_size": DCP_SIZE,
        "dcp_comm_backend": "ag_rs",
        "cp_kv_cache_interleave_size": 1,
        "distributed_executor_backend": "mp",
        "worker_extension_cls": ("vllm.profiler.pcp_dcp_mla.PCPDCPMLAWorkerExtension"),
        "block_size": BLOCK_SIZE,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": 8_192,
        "max_num_seqs": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "compilation_config": {"cudagraph_mode": "NONE"},
        "enable_layerwise_nvtx_tracing": True,
        "attention_config": {"backend": "auto"},
        "disable_log_stats": False,
        "seed": args.seed,
    }
    if args.hf_config_path is not None:
        engine_kwargs["hf_config_path"] = args.hf_config_path
    if capture:
        engine_kwargs["profiler_config"] = {
            "profiler": "cuda",
            "detailed_trace_annotation": True,
        }
    _write_json(output_dir / "requested_engine_config.json", engine_kwargs)
    print("Resolved engine configuration:")
    print(json.dumps(engine_kwargs, indent=2, sort_keys=True))

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(**engine_kwargs)
    hf_config = llm.model_config.hf_config.to_dict()
    _write_json(output_dir / "resolved_hf_config.json", hf_config)
    _validate_hf_config(hf_config)

    worker_metadata = llm.collective_rpc("pcp_dcp_mla_profile_metadata")
    _write_json(output_dir / "worker_metadata.json", worker_metadata)
    _validate_worker_metadata(
        worker_metadata,
        args.expect_outer_backend.upper(),
        args.expected_git_sha,
    )

    vocab_size = int(hf_config["vocab_size"])
    prefix = _make_prefix(prefix_tokens, vocab_size, args.seed)
    suffix_specs = [("warmup_0", 101)]
    measured_count = 1 if smoke else 3
    if capture:
        suffix_specs.append(("nsys_0" if not smoke else "smoke_nsys_0", 301))
    else:
        suffix_specs.extend(
            (f"measure_{index}", 201 + index) for index in range(measured_count)
        )
    suffixes = {
        label: _make_suffix(args.suffix_tokens, vocab_size, args.seed, variant)
        for label, variant in suffix_specs
    }
    first_suffix_tokens = [suffix[0] for suffix in suffixes.values()]
    if len(set(first_suffix_tokens)) != len(first_suffix_tokens):
        raise RuntimeError("Suffixes do not diverge at their first token.")
    token_payload = {"prefix": prefix, "suffixes": suffixes}
    with gzip.open(
        output_dir / "prompt_token_ids.json.gz", "wt", encoding="utf-8"
    ) as file:
        json.dump(token_payload, file, separators=(",", ":"))
    workload = {
        "prefix_tokens": len(prefix),
        "prefix_sha256_le_u32": _token_hash(prefix),
        "suffixes": {
            label: {
                "tokens": len(suffix),
                "first_token": suffix[0],
                "sha256_le_u32": _token_hash(suffix),
            }
            for label, suffix in suffixes.items()
        },
    }
    _write_json(output_dir / "workload.json", workload)

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
        seed=args.seed,
    )
    prime_prompt = TokensPrompt(prompt_token_ids=prefix)
    prime_start = time.perf_counter()
    prime_outputs = llm.generate(prime_prompt, sampling_params, use_tqdm=False)
    prime_wall_ms = (time.perf_counter() - prime_start) * 1_000
    if len(prime_outputs) != 1:
        raise RuntimeError(f"Expected one priming output: {prime_outputs}")
    prime_evidence = _request_evidence(prime_outputs[0], prefix, 0)
    results: dict[str, Any] = {
        "prime": {"frontend_wall_ms": prime_wall_ms, "request": prime_evidence},
        "warmup": None,
        "measured": [],
    }
    _write_json(output_dir / "results.json", results)

    for index, (label, suffix) in enumerate(suffixes.items()):
        prompt_ids = prefix + suffix
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        is_capture_iteration = capture and index == len(suffixes) - 1
        iteration = _run_measured_request(
            llm,
            prompt,
            sampling_params,
            label,
            prefix_tokens,
            is_capture_iteration,
        )
        iteration["profile_validation"] = _validate_iteration_profile(
            iteration["profiles"], prefix_tokens, args.suffix_tokens
        )
        if label == "warmup_0":
            results["warmup"] = iteration
        else:
            results["measured"].append(iteration)
        _write_json(output_dir / "results.json", results)

    results["event_summary"] = _event_summary(results["measured"])
    _write_json(output_dir / "results.json", results)
    _snapshot_patch(output_dir / "local_changes.patch")
    print(json.dumps(results["event_summary"], indent=2, sort_keys=True))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
