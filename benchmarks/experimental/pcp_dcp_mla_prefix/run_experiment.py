# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the dense or sparse MLA PCP+DCP cached-prefix prefill experiment."""

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
os.environ["PATH"] = os.pathsep.join(
    (str(Path(sys.executable).parent), os.environ.get("PATH", ""))
)

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
REQUIRED_SPARSE_DCP_COMMIT = "757c95ce262172fc360d5c65d43b7186a327df70"
SPARSE_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
SPARSE_TOPK_TOKENS = 2_048
SPARSE_INDEX_HEAD_DIM = 128


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


def _validate_repository(expected_sha: str, workload: str) -> dict[str, Any]:
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
    evidence = {
        "experiment_git_sha": head,
        "precompiled_source_sha": expected_sha,
        "precompiled_source_is_ancestor": True,
        "required_pr": "https://github.com/vllm-project/vllm/pull/46570",
        "required_commit": REQUIRED_PCP_COMMIT,
        "required_commit_is_ancestor": True,
    }
    if workload == "sparse":
        sparse_ancestor = _run_command(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                REQUIRED_SPARSE_DCP_COMMIT,
                "HEAD",
            ]
        )
        if sparse_ancestor["returncode"] != 0:
            raise RuntimeError(
                "Sparse workload requires the local PR #46514 overlay commit "
                f"{REQUIRED_SPARSE_DCP_COMMIT}."
            )
        evidence.update(
            {
                "sparse_dcp_pr": "https://github.com/vllm-project/vllm/pull/46514",
                "sparse_dcp_overlay_commit": REQUIRED_SPARSE_DCP_COMMIT,
                "sparse_dcp_overlay_is_ancestor": True,
            }
        )
    return evidence


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
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
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


def _validate_hf_config(config: dict[str, Any], workload: str) -> None:
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
    if workload == "dense" and "index_topk" in config:
        raise RuntimeError("The selected config contains index_topk and is sparse MLA.")
    if workload == "sparse":
        sparse_expected = {
            "index_topk": SPARSE_TOPK_TOKENS,
            "index_head_dim": SPARSE_INDEX_HEAD_DIM,
            "model_type": "deepseek_v32",
        }
        sparse_mismatches = {
            key: {"expected": value, "actual": config.get(key)}
            for key, value in sparse_expected.items()
            if config.get(key) != value
        }
        if sparse_mismatches:
            raise RuntimeError(
                f"DeepSeek-V3.2 sparse geometry mismatch: {sparse_mismatches}"
            )
    if config.get("first_k_dense_replace", 0) <= 0:
        raise RuntimeError("Layer 0 is not configured as a dense MLP layer.")
    if config.get("quantization_config") is not None:
        raise RuntimeError("quantization_config must be None for dummy BF16 weights.")


def _validate_worker_metadata(
    rows: list[dict[str, Any]],
    expect_outer_backend: str,
    expected_source_sha: str,
    workload: str,
) -> None:
    if {row["rank"] for row in rows} != {0, 1}:
        raise RuntimeError(f"Expected ranks 0 and 1, got: {rows}")
    for row in rows:
        if row["world_size"] != 2:
            raise RuntimeError(f"Expected world size 2: {row}")
        if workload == "dense":
            if row["parameter_dtypes"] != ["torch.bfloat16"]:
                raise RuntimeError(f"Weights are not exclusively BF16: {row}")
        else:
            if row["parameter_dtypes"] != ["torch.bfloat16", "torch.float32"]:
                raise RuntimeError(f"Unexpected sparse-model parameter dtypes: {row}")
            expected_fp32_parameters = {
                "model.layers.0.self_attn.indexer.k_norm.weight": 128,
                "model.layers.0.self_attn.indexer.k_norm.bias": 128,
            }
            actual_fp32_parameters = {
                parameter["name"]: parameter["numel"]
                for parameter in row["non_bf16_parameters"]
            }
            fp32_numel = row["parameter_dtype_numel"].get("torch.float32", 0)
            if fp32_numel != 256 or actual_fp32_parameters != expected_fp32_parameters:
                raise RuntimeError(
                    "Sparse dummy model must retain only the two native FP32 "
                    "indexer normalization parameters; got "
                    f"{row['non_bf16_parameters']}"
                )
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
        expected_sparse = workload == "sparse"
        if backend["use_sparse"] != expected_sparse:
            raise RuntimeError(
                f"Expected use_sparse={expected_sparse}, got backend metadata: "
                f"{backend}"
            )
        expected_cache_dtype = "fp8_ds_mla" if expected_sparse else "bfloat16"
        if backend["kv_cache_dtype"] != expected_cache_dtype:
            raise RuntimeError(
                f"Expected {expected_cache_dtype} attention cache: {backend}"
            )
        if expect_outer_backend != "ANY" and (
            backend["outer_backend"] != expect_outer_backend
        ):
            raise RuntimeError(
                "Attention backend selection did not meet the experiment contract: "
                f"expected {expect_outer_backend}, got {backend['outer_backend']}. "
                "Record and resolve the mismatch before accepting the run."
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
    workload: str,
) -> list[dict[str, Any]]:
    expected_local_q = math.ceil(suffix_tokens / (2 * PCP_SIZE)) * 2
    expected_suffix_bytes = (
        suffix_tokens // PCP_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2
    )
    if workload == "sparse":
        expected_suffix_bytes += suffix_tokens // PCP_SIZE * SPARSE_INDEX_HEAD_DIM * 2
    expected_hidden_bytes = suffix_tokens // PCP_SIZE * HIDDEN_SIZE * 2
    sparse_expected_bytes = {
        "sparse_indexer_comm": (expected_local_q * SPARSE_TOPK_TOKENS * 2 * 4),
        "attention_lse_comm": expected_local_q * NUM_ATTENTION_HEADS * 4,
        "attention_output_comm": (
            expected_local_q * NUM_ATTENTION_HEADS * KV_LORA_RANK * 2
        ),
    }
    if {profile["rank"] for profile in profiles} != {0, 1}:
        raise RuntimeError(f"Missing rank profile: {profiles}")

    def validate_category(
        profile: dict[str, Any], category: str, expected_bytes: int
    ) -> dict[str, Any]:
        actual_send = _record_sum(profile, category, "send_bytes")
        actual_recv = _record_sum(profile, category, "recv_bytes")
        if actual_send != expected_bytes or actual_recv != expected_bytes:
            raise RuntimeError(
                f"{category} payload mismatch: expected {expected_bytes}, "
                f"got send={actual_send}, recv={actual_recv}: {profile}"
            )
        return {
            "expected_send_bytes": expected_bytes,
            "actual_send_bytes": actual_send,
            "actual_recv_bytes": actual_recv,
            "collective_count": sum(
                record["category"] == category for record in profile["records"]
            ),
        }

    validation = []
    for profile in profiles:
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

        row: dict[str, Any] = {
            "rank": profile["rank"],
            "user_requests": 1,
            "pcp_virtual_requests": len(
                str(partition_records[0]["fields"]["segment_lengths"]).split("x")
            ),
            "local_q_tokens": expected_local_q,
            "suffix_cache_comm": validate_category(
                profile, "suffix_cache_comm", expected_suffix_bytes
            ),
            "hidden_restore_comm": validate_category(
                profile, "hidden_restore_comm", expected_hidden_bytes
            ),
        }
        if workload == "dense":
            expected_context = _expected_context_comm(
                int(profile["rank"]), prefix_tokens, suffix_tokens
            )
            context = validate_category(
                profile,
                "context_attention_comm",
                expected_context["physical_send_bytes"],
            )
            row["context_attention_comm"] = {**expected_context, **context}
        else:
            if any(
                record["category"] == "context_attention_comm"
                for record in profile["records"]
            ):
                raise RuntimeError("Sparse DSA unexpectedly used dense context gather.")
            for category, expected_bytes in sparse_expected_bytes.items():
                row[category] = validate_category(profile, category, expected_bytes)
            compute_records = [
                record
                for record in profile["records"]
                if record["category"] == "sparse_attention_compute"
            ]
            if len(compute_records) != 1:
                raise RuntimeError(
                    f"Expected one sparse attention kernel range: {compute_records}"
                )
            compute_fields = compute_records[0]["fields"]
            expected_compute_fields = {
                "kernel": "flashmla_fp8_mixed_batch",
                "local_q_tokens": expected_local_q,
                "heads": NUM_ATTENTION_HEADS,
                "topk_tokens": SPARSE_TOPK_TOKENS,
                "cache_dtype": "fp8_ds_mla",
            }
            mismatches = {
                key: {"expected": value, "actual": compute_fields.get(key)}
                for key, value in expected_compute_fields.items()
                if compute_fields.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"Sparse kernel path mismatch: {mismatches}")
            row["sparse_attention_compute"] = {
                **compute_fields,
                "elapsed_ms": compute_records[0]["elapsed_ms"],
            }
            indexer_scope_records = [
                record
                for record in profile["records"]
                if record["category"] == "sparse_indexer_scope"
            ]
            if len(indexer_scope_records) != 1:
                raise RuntimeError(
                    f"Expected one sparse indexer scope: {indexer_scope_records}"
                )
            indexer_fields = indexer_scope_records[0]["fields"]
            expected_indexer_fields = {
                "local_q_tokens": expected_local_q,
                "topk_tokens": SPARSE_TOPK_TOKENS,
                "dcp_world_size": DCP_SIZE,
                "use_pcp": True,
            }
            indexer_mismatches = {
                key: {"expected": value, "actual": indexer_fields.get(key)}
                for key, value in expected_indexer_fields.items()
                if indexer_fields.get(key) != value
            }
            if indexer_mismatches:
                raise RuntimeError(
                    f"Sparse indexer scope mismatch: {indexer_mismatches}"
                )
            row["sparse_indexer_scope"] = indexer_fields
        validation.append(row)
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
        "user_requests": 1,
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
    parser.add_argument("--workload", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--hf-config-path")
    parser.add_argument("--prefix-tokens", type=int)
    parser.add_argument("--suffix-tokens", type=int, default=SUFFIX_TOKENS)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measured-iterations", type=int)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=8_192)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--attention-backend")
    parser.add_argument("--expect-outer-backend")
    parser.add_argument("--allow-non-contract-shape", action="store_true")
    parser.add_argument("--expected-git-sha", default=EXPECTED_GIT_SHA)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sparse = args.workload == "sparse"
    measured_iterations = args.measured_iterations
    if measured_iterations is None:
        measured_iterations = 1 if args.mode.startswith("smoke") else 3
    if args.warmup_iterations < 1 or measured_iterations < 1:
        raise ValueError("Warmup and measured iteration counts must be positive.")
    if args.prefill_chunk_tokens < args.suffix_tokens:
        raise ValueError("Prefill chunk size must fit the measured suffix.")
    model = args.model or (SPARSE_MODEL if sparse else "deepseek-ai/DeepSeek-V3")
    attention_backend = args.attention_backend or (
        "FLASHMLA_SPARSE" if sparse else "FLASHMLA"
    )
    expect_outer_backend = args.expect_outer_backend or attention_backend
    kv_cache_dtype = "fp8_ds_mla" if sparse else "bfloat16"
    repository_evidence = _validate_repository(args.expected_git_sha, args.workload)
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
        "workload": args.workload,
        "model": model,
        "revision": args.revision,
        "hf_config_path": args.hf_config_path,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "warmup_iterations": args.warmup_iterations,
        "measured_iterations": measured_iterations,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "max_model_len": MAX_MODEL_LEN,
        "weight_format": (
            "dummy unquantized BF16 with native sparse FP32 parameters"
            if sparse
            else "dummy unquantized BF16"
        ),
        "activation_dtype": "bfloat16",
        "kv_cache_dtype": kv_cache_dtype,
        "user_requests_per_generate": 1,
        "max_num_seqs": 1,
        "attention_backend_request": attention_backend,
        "mla_prefill_backend_request": "auto",
        "expected_outer_backend": expect_outer_backend,
        "repository_requirement": repository_evidence,
    }
    _write_json(output_dir / "environment.json", metadata)
    _snapshot_patch(output_dir / "local_changes.patch")

    engine_kwargs = {
        "model": model,
        "revision": args.revision,
        "skip_tokenizer_init": True,
        "load_format": "dummy",
        "dtype": "bfloat16",
        "kv_cache_dtype": kv_cache_dtype,
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
        "max_num_batched_tokens": args.prefill_chunk_tokens,
        "max_num_seqs": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "compilation_config": {"cudagraph_mode": "NONE"},
        "enable_layerwise_nvtx_tracing": True,
        "attention_config": {
            "backend": attention_backend,
            "sparse_mla_force_mqa": sparse,
        },
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
    _validate_hf_config(hf_config, args.workload)

    worker_metadata = llm.collective_rpc("pcp_dcp_mla_profile_metadata")
    _write_json(output_dir / "worker_metadata.json", worker_metadata)
    _validate_worker_metadata(
        worker_metadata,
        expect_outer_backend.upper(),
        args.expected_git_sha,
        args.workload,
    )

    vocab_size = int(hf_config["vocab_size"])
    prefix = _make_prefix(prefix_tokens, vocab_size, args.seed)
    suffix_specs = [
        (f"warmup_{index}", 101 + index) for index in range(args.warmup_iterations)
    ]
    if capture:
        suffix_specs.append(("nsys_0" if not smoke else "smoke_nsys_0", 301))
    else:
        suffix_specs.extend(
            (f"measure_{index}", 201 + index) for index in range(measured_iterations)
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
        "workload": args.workload,
        "user_requests_per_generate": 1,
        "max_num_seqs": 1,
        "pcp_virtual_requests_per_rank": 2,
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
        "contract": {
            "workload": args.workload,
            "user_requests_per_generate": 1,
            "max_num_seqs": 1,
            "pcp_virtual_requests_per_rank": 2,
            "warmup_iterations": args.warmup_iterations,
            "measured_iterations": measured_iterations,
        },
        "prime": {"frontend_wall_ms": prime_wall_ms, "request": prime_evidence},
        "warmup": None,
        "warmups": [],
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
            iteration["profiles"],
            prefix_tokens,
            args.suffix_tokens,
            args.workload,
        )
        if label.startswith("warmup_"):
            results["warmup"] = iteration
            results["warmups"].append(iteration)
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
