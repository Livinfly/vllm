"""
Visualize DP load imbalance experiment results.

Reads per-rank JSON files produced by dp_imbalance_baseline.py and generates
four charts:
  1. Per-Rank Total Time (bar chart)
  2. Per-Request E2E Latency (grouped bar, colored by rank)
  3. Prefill Time vs Prompt Length (scatter)
  4. Timeline / Gantt chart (horizontal bars per rank)

Usage:
    python experiments/plot_dp_imbalance.py \
        --input-dir results/extreme \
        --output-dir results/extreme/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(input_dir: str) -> list[dict]:
    """Load all rank_*.json files from the input directory."""
    results = []
    for p in sorted(Path(input_dir).glob("rank_*.json")):
        with open(p) as f:
            results.append(json.load(f))
    if not results:
        raise FileNotFoundError(f"No rank_*.json files found in {input_dir}")
    results.sort(key=lambda r: r["dp_rank"])
    return results


def plot_total_time(results: list[dict], output_dir: Path):
    """Bar chart: total completion time per rank."""
    ranks = [r["dp_rank"] for r in results]
    times = [r["total_time"] for r in results]
    total_tokens = [r["total_prompt_tokens"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(ranks, times, color=plt.cm.Set2(np.linspace(0, 1, len(ranks))))
    ax.set_xlabel("DP Rank")
    ax.set_ylabel("Total Time (s)")
    ax.set_title("Per-Rank Total Completion Time")
    ax.set_xticks(ranks)

    # Annotate with total tokens
    for bar, tok in zip(bars, total_tokens):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{tok} tok", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "1_total_time.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 1_total_time.png")


def plot_e2e_latency(results: list[dict], output_dir: Path):
    """Grouped bar chart: e2e latency per request, colored by rank."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    all_labels = []
    all_latencies = []
    all_colors = []

    for rank_data in results:
        rank = rank_data["dp_rank"]
        for req in rank_data["requests"]:
            label = f"R{rank}:{req['target_prompt_len']}"
            latency = req.get("e2e_latency", req.get("first_token_latency", 0))
            all_labels.append(label)
            all_latencies.append(latency)
            all_colors.append(colors[rank])

    x = np.arange(len(all_labels))
    ax.bar(x, all_latencies, color=all_colors)
    ax.set_xlabel("Request (Rank:PromptLen)")
    ax.set_ylabel("E2E Latency (s)")
    ax.set_title("Per-Request E2E Latency by Rank")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[r["dp_rank"]],
                             label=f"Rank {r['dp_rank']}")
                       for r in results]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    fig.savefig(output_dir / "2_e2e_latency.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 2_e2e_latency.png")


def plot_prefill_vs_length(results: list[dict], output_dir: Path):
    """Scatter plot: prefill time vs prompt length."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for rank_data in results:
        rank = rank_data["dp_rank"]
        lengths = []
        prefill_times = []
        for req in rank_data["requests"]:
            if "prefill_time" in req:
                lengths.append(req["target_prompt_len"])
                prefill_times.append(req["prefill_time"])
        if lengths:
            ax.scatter(lengths, prefill_times, color=colors[rank],
                       label=f"Rank {rank}", s=60, alpha=0.8)

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Prefill Time (s)")
    ax.set_title("Prefill Time vs Prompt Length")
    ax.legend()
    ax.set_xscale("log", base=2)

    fig.tight_layout()
    fig.savefig(output_dir / "3_prefill_vs_length.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 3_prefill_vs_length.png")


def plot_timeline(results: list[dict], output_dir: Path):
    """Gantt-like timeline: horizontal bars showing request execution on each rank."""
    fig, ax = plt.subplots(figsize=(14, max(4, len(results) * 1.5)))
    colors_map = plt.cm.tab20(np.linspace(0, 1, 20))

    y_ticks = []
    y_labels = []

    for rank_data in results:
        rank = rank_data["dp_rank"]
        requests = rank_data["requests"]

        # Sort by scheduled time if available, else by index
        sorted_reqs = sorted(requests,
                             key=lambda r: r.get("scheduled_ts", 0))

        for i, req in enumerate(sorted_reqs):
            y_pos = rank * 1.0
            prompt_len = req["target_prompt_len"]
            color = colors_map[hash(prompt_len) % 20]

            prefill = req.get("prefill_time", 0)
            decode = req.get("decode_time", 0)
            first_token = req.get("first_token_latency", 0)

            # Use first_token_latency as start offset from rank's t=0,
            # then decode time as continuation
            start = first_token - prefill if first_token > prefill else 0

            # Prefill bar
            if prefill > 0:
                ax.barh(y_pos, prefill, left=start, height=0.3,
                        color=color, alpha=0.9,
                        edgecolor="black", linewidth=0.5)
            # Decode bar (lighter)
            if decode > 0:
                ax.barh(y_pos, decode, left=start + prefill, height=0.3,
                        color=color, alpha=0.4,
                        edgecolor="black", linewidth=0.5)

            # Label
            mid = start + (prefill + decode) / 2
            ax.text(mid, y_pos, f"{prompt_len}", ha="center", va="center",
                    fontsize=7, fontweight="bold")

        y_ticks.append(rank * 1.0)
        y_labels.append(f"Rank {rank}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (s)")
    ax.set_title("Request Execution Timeline (dark=prefill, light=decode)")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_dir / "4_timeline.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 4_timeline.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DP imbalance experiment results")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing rank_*.json files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for plots (default: input-dir/plots)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = Path(args.output_dir or f"{input_dir}/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {input_dir}...")
    results = load_results(input_dir)
    print(f"Found {len(results)} ranks")

    # Print summary
    for r in results:
        print(f"  Rank {r['dp_rank']}: {r['num_requests']} requests, "
              f"{r['total_prompt_tokens']} tokens, "
              f"{r['total_time']:.2f}s")

    max_time = max(r["total_time"] for r in results)
    min_time = min(r["total_time"] for r in results)
    print(f"  Imbalance ratio: {max_time / min_time:.2f}x "
          f"(max={max_time:.2f}s, min={min_time:.2f}s)")

    print(f"\nGenerating plots in {output_dir}...")
    plot_total_time(results, output_dir)
    plot_e2e_latency(results, output_dir)
    plot_prefill_vs_length(results, output_dir)
    plot_timeline(results, output_dir)

    print(f"\nDone! All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
