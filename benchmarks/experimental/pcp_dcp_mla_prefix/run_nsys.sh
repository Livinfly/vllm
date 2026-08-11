#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

find_nsys() {
    if [[ -n ${NSYS_BIN:-} && -x ${NSYS_BIN} ]]; then
        printf '%s\n' "$NSYS_BIN"
        return
    fi
    if command -v nsys >/dev/null; then
        command -v nsys
        return
    fi
    local candidate
    for candidate in \
        /opt/nvidia/nsight-systems/*/bin/nsys \
        /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys; do
        if [[ -x $candidate ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    return 1
}

mode=${MODE:-nsys}
case "$mode" in
    nsys)
        measured_label=nsys_0
        default_prefix_tokens=81920
        ;;
    smoke-nsys)
        measured_label=smoke_nsys_0
        default_prefix_tokens=1024
        ;;
    *)
        echo "MODE must be nsys or smoke-nsys" >&2
        exit 2
        ;;
esac

prefix_tokens=${PREFIX_TOKENS:-$default_prefix_tokens}
if [[ ! $prefix_tokens =~ ^[0-9]+$ ]] || \
    (( prefix_tokens <= 0 || prefix_tokens % 2 != 0 )); then
    echo "PREFIX_TOKENS must be a positive even integer" >&2
    exit 2
fi
case "$prefix_tokens" in
    1024) prefix_label=1k ;;
    20480) prefix_label=20k ;;
    40960) prefix_label=40k ;;
    81920) prefix_label=80k ;;
    *) prefix_label=$prefix_tokens ;;
esac
profile_name="prefix${prefix_label}_suffix256"
unique_prefix_send_bytes=$((prefix_tokens * 576))

nsys_bin=$(find_nsys || true)
if [[ -z $nsys_bin ]]; then
    echo "nsys was not found; set NSYS_BIN to its executable path" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    echo ".venv/bin/python is missing; follow AGENTS.md environment setup" >&2
    exit 1
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir=${RUN_DIR:-artifacts/pcp_dcp_mla_prefix/${timestamp}-${mode}}
driver_dir="$run_dir/driver"
profile_prefix="$run_dir/$profile_name"

export NSYS_BIN=$nsys_bin
PATH="$repo_root/.venv/bin:$(dirname "$nsys_bin"):$PATH"
export PATH

profile_help=$($nsys_bin profile --help)
nsys_args=(
    "$nsys_bin" profile
    --trace=cuda,nvtx
    --force-overwrite=true
    --output="$profile_prefix"
)
discard_environment=0
if [[ "$profile_help" == *"--discard-environment"* ]]; then
    nsys_args+=(--discard-environment=true)
    discard_environment=1
fi
if [[ "$profile_help" == *"--trace-fork-before-exec"* ]]; then
    nsys_args+=(--trace-fork-before-exec=true)
fi
if [[ "$profile_help" == *"--capture-range"* ]]; then
    nsys_args+=(--capture-range=cudaProfilerApi)
    if [[ "$profile_help" == *"--capture-range-end"* ]]; then
        nsys_args+=(--capture-range-end=stop)
    fi
else
    echo "Warning: this nsys lacks --capture-range; markers will isolate the target." >&2
fi

driver=(
    .venv/bin/python
    benchmarks/experimental/pcp_dcp_mla_prefix/run_experiment.py
    --mode "$mode"
    --output-dir "$driver_dir"
    --prefix-tokens "$prefix_tokens"
)
if (( prefix_tokens != default_prefix_tokens )); then
    driver+=(--allow-non-contract-shape)
fi
driver+=("$@")

printf 'Resolved command:'
launch=(env -i)
for name in \
    HOME PATH LD_LIBRARY_PATH PYTHONPATH CUDA_VISIBLE_DEVICES CUDA_HOME \
    HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME TMPDIR \
    NCCL_DEBUG NCCL_ALGO NCCL_PROTO LANG LC_ALL; do
    if [[ -n ${!name+x} ]]; then
        launch+=("$name=${!name}")
    fi
done
launch+=(PCP_DCP_MLA_PROFILE=1 VLLM_NVTX_SCOPES_FOR_PROFILING=1)
printf ' %q' "${launch[@]}"
printf ' %q' "${nsys_args[@]}" "${driver[@]}"
printf '\n'
if [[ ${DRY_RUN:-0} == 1 ]]; then
    exit 0
fi
if (( ! discard_environment )); then
    echo "Warning: this nsys lacks --discard-environment; using a clean allowlist." >&2
fi

mkdir -p "$run_dir"
"${launch[@]}" "${nsys_args[@]}" "${driver[@]}"

raw_report="${profile_prefix}.nsys-rep"
if [[ ! -f "$raw_report" ]]; then
    echo "Expected report was not created: $raw_report" >&2
    exit 1
fi

report="${profile_prefix}.sanitized.nsys-rep"
redaction_report="${profile_prefix}.redaction.json"
.venv/bin/python \
    benchmarks/experimental/pcp_dcp_mla_prefix/sanitize_nsys_sqlite.py \
    --sanitize-report "$raw_report" \
    --output "$report" \
    --redaction-report "$redaction_report" \
    --nsys-bin "$nsys_bin" \
    --force

.venv/bin/python \
    benchmarks/experimental/pcp_dcp_mla_prefix/analyze_nsys.py \
    --input "$report" \
    --output-dir "$run_dir/precise-analysis" \
    --label "$measured_label" \
    --unique-prefix-send-bytes "$unique_prefix_send_bytes"

analysis_name="${profile_name}.sanitized"
sqlite="$run_dir/precise-analysis/${analysis_name}.sqlite"
portable_sqlite="$run_dir/precise-analysis/${analysis_name}.portable.sqlite"
.venv/bin/python \
    benchmarks/experimental/pcp_dcp_mla_prefix/sanitize_nsys_sqlite.py \
    --input "$sqlite" \
    --output "$portable_sqlite" \
    --source-report "$report" \
    --force
if [[ ${RUN_OVERVIEW:-1} == 1 ]]; then
    .venv/bin/python \
        tools/profiler/nsys_profile_tools/gputrc2graph.py \
        --in_file "$report,vllm,ds,0" \
        --out_dir "$run_dir/overview" \
        --title "DeepSeek-V3 1L TP1 PCP2 DCP2 prefix${prefix_label} suffix256"
fi

echo "Private raw report: $raw_report"
echo "Publishable sanitized report: $report"
echo "Redaction manifest: $redaction_report"
echo "Portable SQLite: $portable_sqlite"
echo "Results: $run_dir"
