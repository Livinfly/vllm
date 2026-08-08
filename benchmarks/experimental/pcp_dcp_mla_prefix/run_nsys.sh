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
        profile_name=prefix80k_suffix256
        ;;
    smoke-nsys)
        measured_label=smoke_nsys_0
        profile_name=prefix1k_suffix256
        ;;
    *)
        echo "MODE must be nsys or smoke-nsys" >&2
        exit 2
        ;;
esac

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
PATH="$(dirname "$nsys_bin"):$PATH"
export PATH

profile_help=$($nsys_bin profile --help)
nsys_args=(
    "$nsys_bin" profile
    --trace=cuda,nvtx
    --force-overwrite=true
    --output="$profile_prefix"
)
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
)
driver+=("$@")

printf 'Resolved command:'
printf ' %q' env PCP_DCP_MLA_PROFILE=1 VLLM_NVTX_SCOPES_FOR_PROFILING=1
printf ' %q' "${nsys_args[@]}" "${driver[@]}"
printf '\n'
if [[ ${DRY_RUN:-0} == 1 ]]; then
    exit 0
fi

mkdir -p "$run_dir"
env \
    PCP_DCP_MLA_PROFILE=1 \
    VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
    "${nsys_args[@]}" "${driver[@]}"

report="${profile_prefix}.nsys-rep"
if [[ ! -f "$report" ]]; then
    echo "Expected report was not created: $report" >&2
    exit 1
fi

.venv/bin/python \
    benchmarks/experimental/pcp_dcp_mla_prefix/analyze_nsys.py \
    --input "$report" \
    --output-dir "$run_dir/precise-analysis" \
    --label "$measured_label"

if [[ ${RUN_OVERVIEW:-1} == 1 ]]; then
    .venv/bin/python \
        tools/profiler/nsys_profile_tools/gputrc2graph.py \
        --in_file "$report,vllm,ds,0" \
        --out_dir "$run_dir/overview" \
        --title "DeepSeek-V3 1L TP1 PCP2 DCP2 prefix80k suffix256"
fi

echo "Report: $report"
echo "Results: $run_dir"
