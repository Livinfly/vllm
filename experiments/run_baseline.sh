#!/bin/bash
# DP Load Imbalance Baseline Experiments
# Run different distribution presets to measure DP load imbalance.
#
# Usage:
#   bash experiments/run_baseline.sh [preset]
#   bash experiments/run_baseline.sh all        # run all presets
#   bash experiments/run_baseline.sh extreme    # run a single preset
#   bash experiments/run_baseline.sh smoke      # quick smoke test

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# --- Configuration ---
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
DP_SIZE="${DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
LENGTHS="${LENGTHS:-1024,2048,4096,8192,16384,32768}"
NUM_PER_LEN="${NUM_PER_LEN:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEM="${GPU_MEM:-0.9}"
RESULTS_BASE="results"

EXTRA_ARGS=""
if [ -n "$MAX_MODEL_LEN" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max-model-len $MAX_MODEL_LEN"
fi

run_preset() {
    local preset=$1
    local output_dir="${RESULTS_BASE}/${preset}"
    echo "============================================"
    echo "Running preset: $preset"
    echo "  Model: $MODEL"
    echo "  DP: $DP_SIZE, TP: $TP_SIZE"
    echo "  Lengths: $LENGTHS"
    echo "  Output dir: $output_dir"
    echo "============================================"

    python experiments/dp_imbalance_baseline.py \
        --model "$MODEL" \
        --dp-size "$DP_SIZE" \
        --tp-size "$TP_SIZE" \
        --distribution "$preset" \
        --lengths "$LENGTHS" \
        --num-requests-per-length "$NUM_PER_LEN" \
        --output-len "$OUTPUT_LEN" \
        --output-dir "$output_dir" \
        --gpu-memory-utilization "$GPU_MEM" \
        $EXTRA_ARGS

    echo "Generating plots..."
    python experiments/plot_dp_imbalance.py --input-dir "$output_dir"
    echo ""
}

# --- Entry point ---
PRESET="${1:-all}"

case "$PRESET" in
    smoke)
        # Quick smoke test with small model and short lengths
        MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
        LENGTHS="512,1024,2048"
        NUM_PER_LEN=1
        RESULTS_BASE="results/smoke"
        run_preset extreme
        ;;
    all)
        for p in extreme interleaved random uniform; do
            run_preset "$p"
        done
        ;;
    extreme|interleaved|random|uniform)
        run_preset "$PRESET"
        ;;
    *)
        echo "Usage: $0 [smoke|extreme|interleaved|random|uniform|all]"
        exit 1
        ;;
esac

echo "=== All experiments completed ==="
echo "Results in $RESULTS_BASE/"
