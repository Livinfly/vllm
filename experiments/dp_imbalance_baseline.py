"""
DP Imbalance Baseline Experiment

Demonstrates load imbalance across Data Parallel ranks when requests have
varying prompt lengths. Each DP rank runs as an independent process with its
own LLM instance and processes a pre-assigned set of requests.

Usage:
    python experiments/dp_imbalance_baseline.py \
        --model Qwen/Qwen2.5-7B \
        --dp-size 2 \
        --distribution extreme \
        --lengths 1024,2048,4096,8192,16384,32768 \
        --num-requests-per-length 2 \
        --output-len 128 \
        --output-dir results/extreme
"""

import argparse
import json
import os
import random
import time
from multiprocessing import Process
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port


# ---------------------------------------------------------------------------
# Distribution strategies
# ---------------------------------------------------------------------------

def distribute_extreme(lengths: list[int], dp_size: int,
                       num_per_length: int) -> list[list[int]]:
    """Short requests on low ranks, long requests on high ranks."""
    all_lengths = sorted(lengths * num_per_length)
    chunk = len(all_lengths) // dp_size
    remainder = len(all_lengths) % dp_size
    assignments: list[list[int]] = []
    start = 0
    for r in range(dp_size):
        end = start + chunk + (1 if r < remainder else 0)
        assignments.append(all_lengths[start:end])
        start = end
    return assignments


def distribute_interleaved(lengths: list[int], dp_size: int,
                           num_per_length: int) -> list[list[int]]:
    """Round-robin by sorted length index, creating interleaved but unequal."""
    all_lengths = sorted(lengths * num_per_length)
    assignments: list[list[int]] = [[] for _ in range(dp_size)]
    for i, length in enumerate(all_lengths):
        assignments[i % dp_size].append(length)
    return assignments


def distribute_random(lengths: list[int], dp_size: int,
                      num_per_length: int) -> list[list[int]]:
    """Random shuffle then split evenly."""
    all_lengths = lengths * num_per_length
    random.shuffle(all_lengths)
    assignments: list[list[int]] = [[] for _ in range(dp_size)]
    for i, length in enumerate(all_lengths):
        assignments[i % dp_size].append(length)
    return assignments


def distribute_uniform(lengths: list[int], dp_size: int,
                       num_per_length: int) -> list[list[int]]:
    """Greedy bin-packing to balance total tokens per rank."""
    all_lengths = sorted(lengths * num_per_length, reverse=True)
    assignments: list[list[int]] = [[] for _ in range(dp_size)]
    totals = [0] * dp_size
    for length in all_lengths:
        min_rank = totals.index(min(totals))
        assignments[min_rank].append(length)
        totals[min_rank] += length
    return assignments


DISTRIBUTION_FNS = {
    "extreme": distribute_extreme,
    "interleaved": distribute_interleaved,
    "random": distribute_random,
    "uniform": distribute_uniform,
}


def load_custom_distribution(path: str, dp_size: int) -> list[list[int]]:
    """Load a custom distribution from a JSON file.

    Expected format: {"rank_0": [1024, 2048], "rank_1": [8192, 16384], ...}
    """
    with open(path) as f:
        data = json.load(f)
    assignments: list[list[int]] = []
    for r in range(dp_size):
        key = f"rank_{r}"
        if key not in data:
            raise ValueError(f"Custom distribution missing key '{key}'")
        assignments.append(data[key])
    return assignments


# ---------------------------------------------------------------------------
# Synthetic prompt generation
# ---------------------------------------------------------------------------

def generate_prompt_of_length(tokenizer, target_len: int) -> str:
    """Generate a prompt string that tokenizes to approximately *target_len* tokens."""
    # Use a repeating phrase so the token count is predictable
    base = "Hello world this is a test of the data parallel load imbalance experiment. "
    base_ids = tokenizer.encode(base, add_special_tokens=False)
    repeat_count = (target_len // len(base_ids)) + 1
    repeated_ids = (base_ids * repeat_count)[:target_len]
    return tokenizer.decode(repeated_ids)


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def run_rank(
    dp_rank: int,
    dp_size: int,
    dp_master_ip: str,
    dp_master_port: int,
    model: str,
    tp_size: int,
    assigned_lengths: list[int],
    output_len: int,
    output_dir: str,
    distribution: str,
    max_model_len: int | None,
    gpu_memory_utilization: float,
    enforce_eager: bool,
):
    """Entry-point for each DP rank process."""
    # Set DP environment variables expected by vLLM
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    print(f"[Rank {dp_rank}] Starting with {len(assigned_lengths)} requests, "
          f"lengths={assigned_lengths}")

    # Build engine kwargs
    engine_kwargs: dict = dict(
        model=model,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len

    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()

    # Generate synthetic prompts of the desired lengths
    prompts: list[str] = []
    actual_lengths: list[int] = []
    for target_len in assigned_lengths:
        prompt = generate_prompt_of_length(tokenizer, target_len)
        prompts.append(prompt)
        actual_lengths.append(len(tokenizer.encode(prompt)))

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for reproducibility
        max_tokens=output_len,
    )

    # Run inference and measure wall-clock time
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time

    # Collect per-request metrics
    request_results = []
    for i, output in enumerate(outputs):
        metrics = output.metrics
        result = {
            "request_id": f"rank{dp_rank}_req{i}",
            "prompt_len": actual_lengths[i],
            "target_prompt_len": assigned_lengths[i],
            "output_len": len(output.outputs[0].token_ids),
            "generated_text_preview": output.outputs[0].text[:100],
        }
        if metrics is not None:
            result.update({
                "arrival_time": metrics.arrival_time,
                "first_token_latency": metrics.first_token_latency,
                "queued_ts": metrics.queued_ts,
                "scheduled_ts": metrics.scheduled_ts,
                "first_token_ts": metrics.first_token_ts,
                "last_token_ts": metrics.last_token_ts,
                "num_generation_tokens": metrics.num_generation_tokens,
            })
            # Derive timing metrics from timestamps
            if metrics.first_token_ts > 0 and metrics.scheduled_ts > 0:
                result["prefill_time"] = metrics.first_token_ts - metrics.scheduled_ts
            if metrics.last_token_ts > 0 and metrics.first_token_ts > 0:
                result["decode_time"] = metrics.last_token_ts - metrics.first_token_ts
            if metrics.last_token_ts > 0 and metrics.arrival_time > 0:
                # e2e from arrival to last token (note: arrival is wall-clock,
                # last_token is monotonic, so this is approximate)
                result["e2e_latency"] = metrics.first_token_latency + result.get("decode_time", 0)

        request_results.append(result)

    rank_result = {
        "dp_rank": dp_rank,
        "model": model,
        "distribution": distribution,
        "tp_size": tp_size,
        "dp_size": dp_size,
        "output_len_setting": output_len,
        "total_time": total_time,
        "num_requests": len(assigned_lengths),
        "total_prompt_tokens": sum(actual_lengths),
        "requests": request_results,
    }

    # Save JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"rank_{dp_rank}.json"
    with open(result_file, "w") as f:
        json.dump(rank_result, f, indent=2)
    print(f"[Rank {dp_rank}] Done. Total time: {total_time:.2f}s. "
          f"Results saved to {result_file}")

    # Give engine time to clean up
    time.sleep(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DP Load Imbalance Baseline Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Model name or path")
    parser.add_argument("--dp-size", type=int, default=2,
                        help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--distribution", type=str, default="extreme",
                        choices=["extreme", "interleaved", "random",
                                 "uniform", "custom"],
                        help="Request distribution preset")
    parser.add_argument("--custom-distribution-file", type=str, default=None,
                        help="JSON file for custom distribution")
    parser.add_argument("--lengths", type=str,
                        default="1024,2048,4096,8192,16384,32768",
                        help="Comma-separated list of prompt lengths")
    parser.add_argument("--num-requests-per-length", type=int, default=2,
                        help="Number of requests per prompt length")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Output tokens per request (kept constant)")
    parser.add_argument("--output-dir", type=str, default="results/baseline",
                        help="Directory to save JSON results")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max model context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization ratio")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graph to save memory")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Seconds before killing unresponsive process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    lengths = [int(x) for x in args.lengths.split(",")]
    dp_size = args.dp_size

    print(f"=== DP Imbalance Baseline ===")
    print(f"Model: {args.model}")
    print(f"DP size: {dp_size}, TP size: {args.tp_size}")
    print(f"Distribution: {args.distribution}")
    print(f"Lengths: {lengths}")
    print(f"Requests per length: {args.num_requests_per_length}")
    print(f"Output len: {args.output_len}")

    # Generate assignments
    if args.distribution == "custom":
        if not args.custom_distribution_file:
            raise ValueError("--custom-distribution-file required "
                             "when --distribution=custom")
        assignments = load_custom_distribution(
            args.custom_distribution_file, dp_size)
    else:
        assignments = DISTRIBUTION_FNS[args.distribution](
            lengths, dp_size, args.num_requests_per_length)

    for r in range(dp_size):
        total_tokens = sum(assignments[r])
        print(f"  Rank {r}: {len(assignments[r])} requests, "
              f"lengths={sorted(assignments[r])}, "
              f"total_tokens={total_tokens}")

    # Save experiment config
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model,
        "dp_size": dp_size,
        "tp_size": args.tp_size,
        "distribution": args.distribution,
        "lengths": lengths,
        "num_requests_per_length": args.num_requests_per_length,
        "output_len": args.output_len,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "assignments": {f"rank_{r}": assignments[r] for r in range(dp_size)},
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Launch DP rank processes
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    procs = []
    for rank in range(dp_size):
        proc = Process(
            target=run_rank,
            args=(
                rank,
                dp_size,
                dp_master_ip,
                dp_master_port,
                args.model,
                args.tp_size,
                assignments[rank],
                args.output_len,
                args.output_dir,
                args.distribution,
                args.max_model_len,
                args.gpu_memory_utilization,
                args.enforce_eager,
            ),
        )
        proc.start()
        procs.append(proc)
        print(f"  Started rank {rank} process (PID {proc.pid})")

    # Wait for all ranks to finish
    exit_code = 0
    for rank, proc in enumerate(procs):
        proc.join(timeout=args.timeout)
        if proc.exitcode is None:
            print(f"[ERROR] Rank {rank} (PID {proc.pid}) timed out, killing.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode != 0:
            print(f"[ERROR] Rank {rank} exited with code {proc.exitcode}")
            exit_code = proc.exitcode

    if exit_code == 0:
        print(f"\n=== All ranks completed. Results in {args.output_dir}/ ===")
        print(f"Run visualization: python experiments/plot_dp_imbalance.py "
              f"--input-dir {args.output_dir}")
    exit(exit_code)


if __name__ == "__main__":
    main()
