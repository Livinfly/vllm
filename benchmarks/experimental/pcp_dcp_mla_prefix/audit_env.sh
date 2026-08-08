#!/usr/bin/env bash
set -u

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

run() {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

run git status --short --branch
run git rev-parse HEAD
run git rev-parse origin/main
run git show -s --format=fuller HEAD
run command -v uv
run command -v nsys
nsys_bin=$(find_nsys || true)
if [[ -n $nsys_bin ]]; then
    run "$nsys_bin" --version
else
    printf '\nnsys was not found; set NSYS_BIN to its executable path\n'
fi
run nvidia-smi
run nvidia-smi topo -m
run nvidia-smi \
    --query-gpu=index,name,uuid,pci.bus_id,driver_version,pstate,clocks.sm,clocks.mem,power.draw,power.limit,memory.total \
    --format=csv,noheader,nounits
run test -x .venv/bin/python
run .venv/bin/python -V
run .venv/bin/python -c 'import pandas, plotly, regex'
