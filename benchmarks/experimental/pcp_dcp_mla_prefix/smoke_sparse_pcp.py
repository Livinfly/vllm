# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a small H100 sparse-MLA PCP+DCP correctness smoke test."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PATH"] = os.pathsep.join(
    (str(Path(sys.executable).parent), os.environ.get("PATH", ""))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-tokens", type=int, default=1_024)
    parser.add_argument("--suffix-tokens", type=int, default=256)
    parser.add_argument("--pcp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dcp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--interleave-size", type=int, choices=(1, 64), default=1)
    parser.add_argument(
        "--history-mode",
        choices=("all_gather", "cuda_vmm"),
        default="all_gather",
    )
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def make_tokens(length: int, offset: int) -> list[int]:
    return [512 + (offset + index * 48_271) % 128_000 for index in range(length)]


def main() -> None:
    args = parse_args()
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be non-negative")
    if args.history_mode == "cuda_vmm" and (args.pcp_size, args.dcp_size) != (2, 2):
        raise ValueError("cuda_vmm history mode requires PCP=2 and DCP=2")

    block_size = 64
    if (
        args.history_mode == "cuda_vmm"
        and args.prefix_tokens % (block_size * args.dcp_size) != 0
    ):
        raise ValueError(
            "cuda_vmm prefix-cache experiments require a full virtual-block prefix hit"
        )

    use_cuda_vmm = args.history_mode == "cuda_vmm"
    os.environ["VLLM_USE_PCP_OWNER_HISTORY"] = "1" if use_cuda_vmm else "0"
    if use_cuda_vmm:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        os.environ["VLLM_PCP_OWNER_PREFILL_MODE"] = "direct"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    max_model_len = max(4_096, args.prefix_tokens + args.suffix_tokens + 1)
    model_kwargs: dict[str, object] = {}
    if use_cuda_vmm:
        model_kwargs["model_class_overrides"] = {
            "DeepseekV32ForCausalLM": (
                "vllm.models.deepseek_v32.nvidia.model:DeepseekV32ForCausalLM"
            )
        }
    llm = LLM(
        model="deepseek-ai/DeepSeek-V3.2-Exp",
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        kv_cache_dtype="fp8_ds_mla",
        hf_overrides={
            "num_hidden_layers": 1,
            "num_nextn_predict_layers": 0,
            "quantization_config": None,
        },
        tensor_parallel_size=1,
        prefill_context_parallel_size=args.pcp_size,
        decode_context_parallel_size=args.dcp_size,
        dcp_comm_backend="ag_rs",
        cp_kv_cache_interleave_size=args.interleave_size,
        distributed_executor_backend="mp",
        block_size=block_size,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=2_048,
        max_num_seqs=1,
        gpu_memory_utilization=0.2,
        enforce_eager=True,
        compilation_config={"cudagraph_mode": "NONE"},
        kernel_config={"enable_flashinfer_autotune": False},
        attention_config={
            "backend": "FLASHMLA_SPARSE",
            "sparse_mla_force_mqa": True,
        },
        profiler_config={"profiler": "cuda"} if args.profile else None,
        disable_log_stats=True,
        seed=17,
        **model_kwargs,
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
        seed=17,
        logprobs=20,
    )
    prefix = make_tokens(args.prefix_tokens, 17)
    suffix = make_tokens(args.suffix_tokens, 101)

    prime = llm.generate(
        TokensPrompt(prompt_token_ids=prefix), sampling, use_tqdm=False
    )[0]
    warmup_cached_tokens = []
    for iteration in range(args.warmup_iterations):
        warmup_suffix = make_tokens(args.suffix_tokens, 201 + iteration)
        warmup = llm.generate(
            TokensPrompt(prompt_token_ids=prefix + warmup_suffix),
            sampling,
            use_tqdm=False,
        )[0]
        warmup_cached_tokens.append(warmup.num_cached_tokens)

    if args.profile:
        llm.start_profile("sparse_pcp_80k_256")
    start = time.perf_counter()
    try:
        measured = llm.generate(
            TokensPrompt(prompt_token_ids=prefix + suffix), sampling, use_tqdm=False
        )[0]
    finally:
        elapsed_seconds = time.perf_counter() - start
        if args.profile:
            llm.stop_profile()
    sampled = measured.outputs[0]
    assert sampled.logprobs is not None
    token_logprobs = sampled.logprobs[0]
    assert token_logprobs is not None
    measured_computed_tokens = len(prefix) + len(suffix) - measured.num_cached_tokens
    result = {
        "history_mode": args.history_mode,
        "pcp_size": args.pcp_size,
        "dcp_size": args.dcp_size,
        "interleave_size": args.interleave_size,
        "prime_cached_tokens": prime.num_cached_tokens,
        "warmup_cached_tokens": warmup_cached_tokens,
        "measured_cached_tokens": measured.num_cached_tokens,
        "measured_elapsed_seconds": elapsed_seconds,
        "measured_computed_tokens": measured_computed_tokens,
        "expected_local_suffix_tokens": len(suffix) // args.pcp_size,
        "sampled_token_id": sampled.token_ids[0],
        "top_logprobs": {
            str(token_id): value.logprob
            for token_id, value in sorted(
                token_logprobs.items(),
                key=lambda item: item[1].rank or 0,
            )
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if measured.num_cached_tokens != len(prefix):
        raise RuntimeError(f"Prefix cache mismatch: {result}")
    if measured_computed_tokens != len(suffix):
        raise RuntimeError(f"Computed suffix mismatch: {result}")
    if any(tokens != len(prefix) for tokens in warmup_cached_tokens):
        raise RuntimeError(f"Warmup prefix cache mismatch: {result}")


if __name__ == "__main__":
    main()
