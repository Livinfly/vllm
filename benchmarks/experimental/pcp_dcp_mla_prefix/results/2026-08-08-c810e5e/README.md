# DeepSeek-V3 1L PCP2+DCP2 prefix-prefill results

## Outcome

The experiment completed on 2026-08-08 UTC. The ordinary run contains one
warmup and three measured suffixes. The Nsight Systems run contains another
warmup and exactly one separately captured suffix. Every suffix request was an
81,920-token cache hit followed by exactly 256 computed tokens in one scheduler
step.

The headline attribution result is the slower rank (rank 0) from the eager nsys
capture:

| Scope | T (ms) | C (ms) | A (ms) | O (ms) | E (ms) | Raw comm | Overlap | Exposed comm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `self_attn` | 22.673216 | 1.562842 | 20.417959 | 0 | 1.562842 | 6.8929% | 0% | 6.8929% |
| `full_layer` | 23.198591 | 1.562842 | 20.719494 | 0 | 1.562842 | 6.7368% | 0% | 6.7368% |

`A` is the union of non-NCCL GPU kernels in the selected scope. It therefore
retains projection/cache/GEMM work rather than counting only the FlashAttention
kernel. `E` is exposed NCCL timeline, not a causal-delay claim.

## Frozen environment

| Item | Value |
| --- | --- |
| GPUs | 2x NVIDIA H100 80GB HBM3, NV18 |
| Driver / reported CUDA | 550.144.03 / 12.4 |
| Local CUDA toolkit | 12.8.93, not used to build vLLM |
| PyTorch / CUDA runtime | 2.13.0+cu129 / 12.9 |
| NCCL | 2.29.7 |
| Nsight Systems | 2025.1.1.0 |
| Python | 3.12.3 through `.venv/bin/python` |
| Precompiled source SHA | `c810e5ee9976ad86b81d1277b53e76d0ee639414` |
| Experiment-code SHA | `feb755ceb3442c37208ba8a451e65f588b1e007f` |
| `origin/main` at run | `e644c8cd8c734bf3b5e662a4bc363cfa8524d821` |
| Required PCP commit | `b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd` |
| vLLM wheel | `0.26.1rc1.dev457+gc810e5ee9.d20260808.precompiled` |

The frozen source contains the merge commit from PR #46570. Native vLLM
compilation was not required: the environment used the exact-commit `cu129`
precompiled wheel with BF16 dummy, unquantized weights. No vLLM CUDA/C++ source
was built with nvcc, CMake, or Ninja. FlashInfer printed its runtime autotuner
messages during startup, before the measured capture; the inference JIT monitor
reported no measured-path compilation.

The model has one dense layer-0-shaped DeepSeek-V3 decoder layer, hidden size
7,168, 128 attention heads, MLA ranks/dimensions 512+128+64/128, BF16
activations and KV cache, TP=1, PCP=2, DCP=2, `ag_rs`, interleave 1, block size
64, eager execution, and no CUDA graphs. The scheduler used
`max_num_seqs=1`, `max_num_batched_tokens=8192`, and `max_model_len=131072`.

Backend selection remained `auto`. Both worker objects report outer
`FLASH_ATTN_MLA` with prefill backend `FLASH_ATTN`; the trace contains SM90
`flash::FlashAttnFwdSm90` kernels. This differs from the handoff's expected
outer `FLASHMLA`, and no backend was forced to hide that difference.

Full machine, topology, clock/power, config, and worker records are in
[events/environment.json](events/environment.json),
[events/resolved_hf_config.json](events/resolved_hf_config.json), and
[events/worker_metadata.json](events/worker_metadata.json).
The post-code-commit analyzer/validation diff is preserved in
[final-local-changes.patch.gz](final-local-changes.patch.gz).

## Cache-hit and workload evidence

The priming request created exactly 81,920 cache tokens. Each warmup/measured
request reports all of the following:

- `prompt_tokens=82176`;
- `num_cached_tokens=81920`;
- `num_computed_prompt_tokens=256`;
- `num_cache_creation_tokens=256`;
- one sampled token and a finished request.

The prefix hash is
`75471f48427567ae6bd7c56b3a48c8c07cfc5eed29883046eb9bb20b1fe4cfc5`.
The ordinary warmup/measured suffix first tokens are 10,309, 20,009, 20,106,
and 20,203. The nsys suffix first token is 29,709. Their complete hashes differ
and the exact token IDs are retained in the two `prompt_token_ids.json.gz`
files. See [events/results.json](events/results.json),
[events/workload.json](events/workload.json), and
[nsys/driver/results.json](nsys/driver/results.json).

PCP partition markers prove 128 local Q tokens per rank. Rank 0 received slices
`0:64,192:256`; rank 1 received `64:128,128:192`.

## Ordinary CUDA-event timing

| Iteration | self rank 0 (ms) | self rank 1 (ms) | layer rank 0 (ms) | layer rank 1 (ms) |
| --- | ---: | ---: | ---: | ---: |
| `measure_0` | 22.065216 | 22.275072 | 22.494240 | 22.668064 |
| `measure_1` | 21.762432 | 22.030497 | 22.155329 | 22.424831 |
| `measure_2` | 23.337088 | 21.796127 | 23.732096 | 22.202047 |

Step latency uses the maximum rank for each iteration:

| Scope | Mean (ms) | p50 (ms) | p90 (ms) | Stdev (ms) | CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `self_attn` | 22.547552 | 22.275072 | 23.124685 | 0.694607 | 3.08% |
| `full_layer` | 22.941664 | 22.668064 | 23.519289 | 0.695254 | 3.03% |

The first two iterations are rank-1 limited; the third is rank-0 limited and
accounts for most of the observed spread. With only three samples, p90 is an
interpolation rather than a tail estimate. The nsys run's independent CUDA
events were 22.836063 ms for `self_attn` and 23.301344 ms for `full_layer`,
1.28% and 1.57% above the ordinary means.

## Per-rank nsys interval metrics

All values below are interval unions except the explicitly named raw sums.

| Rank | Scope | T (ms) | C (ms) | A (ms) | O (ms) | E (ms) | Raw comm | Overlap | Exposed comm | Raw compute sum (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `self_attn` | 22.673216 | 1.562842 | 20.417959 | 0 | 1.562842 | 6.8929% | 0% | 6.8929% | 20.422471 |
| 0 | `full_layer` | 23.198591 | 1.562842 | 20.719494 | 0 | 1.562842 | 6.7368% | 0% | 6.7368% | 20.724006 |
| 1 | `self_attn` | 22.137656 | 0.587679 | 20.214139 | 0 | 0.587679 | 2.6547% | 0% | 2.6547% | 20.216891 |
| 1 | `full_layer` | 22.667575 | 0.587679 | 20.514619 | 0 | 0.587679 | 2.5926% | 0% | 2.5926% | 20.517371 |

The attention communication raw sums equal their unions in this capture. The
complete per-device/per-stream records and kernel names are in
[nsys/precise-analysis/kernels.csv](nsys/precise-analysis/kernels.csv).

## Communication attribution

| Rank | Category | Collectives | Union/raw (us) | Send bytes | Receive bytes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | DCP context | 3 | 835.196 | 94,482,432 | 94,482,432 |
| 0 | PCP suffix cache | 2 | 727.646 | 147,456 | 147,456 |
| 0 | Attention path total | 5 | 1,562.842 | 94,629,888 | 94,629,888 |
| 0 | Hidden restore | 1 | 27.840 | 1,835,008 | 1,835,008 |
| 1 | DCP context | 3 | 573.791 | 94,482,432 | 94,482,432 |
| 1 | PCP suffix cache | 2 | 13.888 | 147,456 | 147,456 |
| 1 | Attention path total | 5 | 587.679 | 94,629,888 | 94,629,888 |
| 1 | Hidden restore | 1 | 79.552 | 1,835,008 | 1,835,008 |

Hidden restore is outside both scope denominators. The rank-asymmetric NCCL
kernel residency, especially the rank-0 suffix gathers, reflects launch/wait
skew visible in the GPU timeline; these durations are not summed across ranks.

The handoff's one-copy prefix baseline is 47,185,920 bytes per rank. PR #46570's
DualChunkSwap exposes two virtual contexts on each rank, so the implementation
physically gathers 82,016 local rows, or 94,482,432 bytes: a 2.00234375x
amplification that includes 96 local rows of preceding suffix context. The three
physical gather chunks carry 37,748,736, 37,748,736, and 18,984,960 bytes.

| Rank | Physical-payload bandwidth (GB/s) | Unique-prefix baseline bandwidth (GB/s) |
| ---: | ---: | ---: |
| 0 | 113.126 | 56.497 |
| 1 | 164.663 | 82.235 |

Both bandwidths use the same per-rank DCP union duration. The first uses actual
physical bytes; the second follows the handoff's required 47,185,920-byte
numerator.

## Reproduction

The native environment was installed without a local vLLM build:

```bash
VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=c810e5ee9976ad86b81d1277b53e76d0ee639414 \
  VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
  uv pip install -e . --torch-backend=cu129 --no-build-isolation
```

Ordinary run:

```bash
.venv/bin/python \
  benchmarks/experimental/pcp_dcp_mla_prefix/run_experiment.py \
  --mode events \
  --output-dir artifacts/pcp-dcp-events-80k
```

The final nsys capture used the repository launcher and removed credential-like
variables from the target process environment. The equivalent current launcher
invocation is:

```bash
ALLOW_NSYS_ENV_CAPTURE=1 MODE=nsys \
  RUN_DIR=artifacts/pcp-dcp-nsys-80k-sanitized \
  bash benchmarks/experimental/pcp_dcp_mla_prefix/run_nsys.sh
```

The explicit opt-in is required because the installed nsys 2025.1.1 lacks
`--discard-environment`. A current nsys that supports that option does not need
the opt-in. Use `DRY_RUN=1` to print the fully resolved command before launch.

Re-run precise analysis directly from the committed portable trace:

```bash
.venv/bin/python \
  benchmarks/experimental/pcp_dcp_mla_prefix/analyze_nsys.py \
  --input benchmarks/experimental/pcp_dcp_mla_prefix/results/\
2026-08-08-c810e5e/nsys/precise-analysis/\
prefix80k_suffix256.portable.sqlite \
  --output-dir artifacts/pcp-dcp-reanalysis \
  --label nsys_0 --unique-prefix-send-bytes 47185920
```

The portable SQLite is
[nsys/precise-analysis/prefix80k_suffix256.portable.sqlite](nsys/precise-analysis/prefix80k_suffix256.portable.sqlite),
with SHA-256
`6091775e615a9c599a5ad263d21c811a0c5295e1fdfb8971b3330d4b09ed1996`.
It records the local source report SHA-256
`dbf19bda6437cc6583ae679df959ec2b1a151f5c7f2ac37d7110249fdc222a29`
in its metadata table. The source report remains in local `artifacts/` because
nsys 2025.1.1 retained a host credential value even though the target process
environment was scrubbed. It was deliberately excluded from Git history. The
portable database contains only the 18 experiment NVTX markers, 686 CUDA
runtime rows, 178 GPU kernel rows, and the 86 kernel/runtime names they use;
reanalysis reproduces the checked-in summary exactly. Raw GPU trace CSVs are
also retained for independent inspection.

The nsys driver `results.json` SHA-256 is
`10b5a52b118e222a1fe88d608d5d076dc97528407c37a1da8c4a813ff3aa35b0`.
The ordinary `results.json` SHA-256 is
`671059217e50ab608efb692469f025ca2df9600ef55e3fe625589a2d02108a0f`.

## Limitations

- This is attribution-oriented eager/NVTX execution, not production graph mode.
- Dummy BF16 weights avoid a checkpoint but make the result dtype- and
  loader-specific.
- Layer 0 is deliberately dense; it is not representative of an average
  DeepSeek-V3 MoE layer.
- Auto backend selection produced FlashAttention MLA rather than the handoff's
  anticipated FlashMLA path.
- Three ordinary repetitions characterize short-run stability but not tails.
- NCCL timeline exposure does not prove equivalent end-to-end causal delay.
- The original `.nsys-rep` cannot be published safely with nsys 2025.1.1 on
  this node. The allowlisted portable SQLite preserves every field consumed by
  the analyzer, but not unrelated Nsight process/host metadata or GUI features.
- The coarse `gputrc2graph.py` output is retained only as a sanity check; the
  reported communication ratios come from the rank-aware NVTX analyzer.
