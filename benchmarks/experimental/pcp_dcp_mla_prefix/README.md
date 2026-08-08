# Dense MLA cached-prefix PCP+DCP experiment

This directory contains an attribution-oriented experiment for one
DeepSeek-V3 layer with TP=1, PCP=2, and DCP=2 on two H100s. It uses dummy,
unquantized BF16 weights, BF16 activations, a BF16 KV cache, eager execution,
and automatic attention-backend selection.

The default run is frozen to `c810e5ee9976ad86b81d1277b53e76d0ee639414`,
the source commit matching the available `cu129` precompiled wheel. The driver
also verifies that PR #46570's merge commit
`b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd` is an ancestor. Override
`--expected-git-sha` only when deliberately moving to another matching wheel.

The driver fails closed unless all of the following are observed:

- one layer with DeepSeek-V3's original dimensions and a dense layer-0 MLP;
- no `index_topk` and no weight quantization;
- two worker ranks and the requested TP/PCP/DCP configuration;
- exactly 81,920 cached prompt tokens and 256 computed suffix tokens;
- one scheduler/model step for every suffix request;
- about 128 local Q tokens on each PCP rank;
- the expected per-rank DCP, PCP-cache, and hidden-restore payloads;
- automatic outer-backend selection of `FLASH_ATTN_MLA` on the frozen commit.

The last check does not force a backend. On the frozen precompiled commit, H100
selects outer `FLASH_ATTN_MLA` and prefill `FLASH_ATTN`, even though the handoff
expected outer `FLASHMLA`. The run records this as an environment/version
deviation instead of forcing a different path. `--expect-outer-backend ANY` is
available only for diagnosis.

## Environment

Follow the repository `AGENTS.md`; do not use system Python or bare pip.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install
VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=c810e5ee9976ad86b81d1277b53e76d0ee639414 \
  VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
  uv pip install -e . --torch-backend=cu129
```

These variables select prebuilt native extensions; no local CUDA/C++ build is
needed. On a workspace with a small `uv` cache quota, install the `cu129` Torch
packages first and add `--no-build-isolation` to the final editable install.

Nsight Systems must be installed by the node image or administrator. The
launcher checks `NSYS_BIN`, `PATH`, and common NVIDIA installation directories,
in that order. Audit the node before a run:

```bash
benchmarks/experimental/pcp_dcp_mla_prefix/audit_env.sh
```

The model defaults to `deepseek-ai/DeepSeek-V3`. Only its configuration is
needed because the dummy loader constructs one BF16 layer. Use `--revision` or
`--hf-config-path` to pin an already downloaded configuration.

## Proposed commands

First inspect the exact nsys launch without executing it:

```bash
DRY_RUN=1 MODE=smoke-nsys \
  benchmarks/experimental/pcp_dcp_mla_prefix/run_nsys.sh
```

Run the 1K-prefix smoke capture and verify both ranks, prefix reuse, all three
communication categories, CUDA-profiler capture, and SQLite parsing:

```bash
MODE=smoke-nsys RUN_DIR=artifacts/pcp-dcp-smoke \
  benchmarks/experimental/pcp_dcp_mla_prefix/run_nsys.sh
```

Run the ordinary CUDA-event experiment (one warmup and three measured suffixes):

```bash
PCP_DCP_MLA_PROFILE=1 VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
  .venv/bin/python \
  benchmarks/experimental/pcp_dcp_mla_prefix/run_experiment.py \
  --mode events \
  --output-dir artifacts/pcp-dcp-events
```

After the smoke succeeds, capture one additional 80K nsys iteration:

```bash
MODE=nsys RUN_DIR=artifacts/pcp-dcp-nsys \
  benchmarks/experimental/pcp_dcp_mla_prefix/run_nsys.sh
```

The launcher checks the installed `nsys profile --help`. It enables child
process tracing and `cudaProfilerApi` capture only when those flags exist. The
driver primes and warms the cache before calling `cudaProfilerStart`, then
profiles exactly one distinct suffix request.

## Workload and evidence

The driver submits `TokensPrompt(prompt_token_ids=...)` directly. It writes the
exact prefix and every suffix to `prompt_token_ids.json.gz`, plus hashes and
first-token divergence evidence to `workload.json`. Requests are sequential and
use `SamplingParams(max_tokens=1)`.

Every measured `RequestOutput.num_cached_tokens` must equal the prefix length.
For the contract run this proves an 81,920-token cache hit, while the returned
prompt IDs prove the measured prompt is exactly 82,176 tokens. Instrumentation
records one PCP partition marker; its two segment lengths must sum to 128 local
Q tokens.

The BF16 payloads per rank are:

| Category | Send bytes | Receive bytes |
| --- | ---: | ---: |
| Unique 81,920-token DCP prefix baseline | 47,185,920 | 47,185,920 |
| Actual DCP context gather | 94,482,432 | 94,482,432 |
| PCP suffix latent gathers | 147,456 | 147,456 |
| PCP hidden restore | 1,835,008 | 1,835,008 |

The first DCP row is the handoff's one-copy theoretical prefix payload. PR #46570's
DualChunkSwap represents each rank's two Q chunks as two virtual
requests. For rank 0 their context lengths are 81,920 and 82,112; for rank 1
they are 81,984 and 82,048. Both therefore gather 82,016 local rows at DCP=2:
`82,016 * 576 * 2 = 94,482,432` bytes. This is 2.00234375 times the unique-prefix
baseline and includes 96 local rows of earlier suffix context. The validator
checks this physical implementation volume and preserves the one-copy baseline
separately so the amplification remains visible.

The source annotations carry actual tensor width, element size, local token
count, world size, logical send/receive bytes, and context chunk index. A shape
or payload mismatch aborts the run.

The temporary patch also sizes V2's synthetic startup prefill to at least
`2 * PCP`. With the upstream two-token warmup, PCP=2 gives rank 0 no context:
rank 1 enters a DCP context all-gather while rank 0 advances to hidden restore,
deadlocking startup. Four warmup tokens keep the collective sequence aligned;
measured prompts and timing ranges are unchanged.

## Precise trace analysis

The launcher invokes `analyze_nsys.py`. It can also be run directly:

```bash
.venv/bin/python \
  benchmarks/experimental/pcp_dcp_mla_prefix/analyze_nsys.py \
  --input artifacts/pcp-dcp-nsys/prefix80k_suffix256.nsys-rep \
  --output-dir artifacts/pcp-dcp-nsys/precise-analysis \
  --label nsys_0
```

The analyzer exports SQLite, resolves NVTX ranges, joins CUDA runtime launches
to GPU kernels by process and correlation ID, and preserves device, PID/TID,
stream, start, end, and kernel name. Communication kernels are attributed by
their narrowly scoped source marker and checked against the configurable NCCL
name regex. Unclassified NCCL work inside either attention scope is an error.

For each rank and each of `self_attn` and `full_layer`, it computes interval
unions before reporting:

```text
T = scope GPU wall span
C = union(context_attention_comm, suffix_cache_comm)
A = union(non-NCCL kernels in the scope)
O = intersection(C, A)
E = C - O
```

It emits `summary.json`, `summary.csv`, `experiment_ranges.csv`, and
`kernels.csv`. The headline is the slower rank. Raw summed kernel durations are
retained as secondary diagnostics; hidden restore remains outside both scope
denominators. The prefix table also reports send-only effective bandwidth.

Run the analyzer's CPU-only interval/schema smoke test with:

```bash
.venv/bin/python \
  benchmarks/experimental/pcp_dcp_mla_prefix/analyze_nsys.py \
  --self-test
```

Each driver result directory contains `local_changes.patch`, environment and
worker metadata, the fully resolved HF configuration, the exact requested
engine configuration, cache-hit evidence, CUDA-event samples, and mean/p50/p90
max-rank summaries. This is experiment-only instrumentation committed to the
dedicated analysis branch; it is not intended as an upstream PR.
