# FlashMLA DeepSeek-V3 1L PCP2+DCP2 prefix-prefill results

## Outcome

The experiment completed on 2026-08-11 UTC for 20K, 40K, and 80K cached
prefixes, each followed by a 256-token computed suffix. Every ordinary run used
one warmup and three measured suffixes. Every Nsight Systems run used another
warmup and exactly one separately captured suffix. After the initial sweep,
20K was captured three more times to test the stability of its attribution.

Both workers resolved the explicitly requested outer backend to `FLASHMLA`
with `FlashMLAImpl`; its prefill backend was `FLASH_ATTN`. PCP and dense
attention were enabled, sparse attention was disabled, and every request had
the exact requested cache hit in one scheduler/model step.

The outer-backend name does not mean that a FlashMLA decode kernel ran in this
workload. PCP keeps the 256-token cached-prefix extension on the common MHA
prefill path, which calls `FlashAttnPrefillBackend` for both outer backends. The
traces consequently contain FlashAttention SM90 prefill kernels and no
FlashMLA dense decode kernel. This result measures the PCP+DCP prefix-prefill
path selected under a FlashMLA outer backend; it is not a direct comparison of
the two backends' decode kernels.

## Frozen environment

| Item | Value |
| --- | --- |
| GPUs | 2x NVIDIA H100 80GB HBM3, NV18 |
| Driver / reported CUDA | 570.195.03 / 12.8 |
| Local CUDA toolkit | 12.8.93, not used to build vLLM |
| PyTorch / CUDA runtime | 2.13.0+cu129 / 12.9 |
| NCCL | 2.29.7 |
| Nsight Systems | 2025.1.1.0 |
| Python | 3.12.3 through `.venv/bin/python` |
| Precompiled source SHA | `c810e5ee9976ad86b81d1277b53e76d0ee639414` |
| Experiment-code HEAD | `14e4fa7a97866ebf156bdb07192dfdff31458252` plus the recorded local patch |
| 20K repeat-capture HEAD | `bc0b57a94041d3f7bf770a4977462466cd07bb1a` |
| `origin/main` at run | `e644c8cd8c734bf3b5e662a4bc363cfa8524d821` |
| Required PCP commit | `b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd` |
| vLLM wheel | `0.26.1rc1.dev457+gc810e5ee9.d20260808.precompiled` |

The model is one dense layer-0-shaped DeepSeek-V3 decoder layer with dummy,
unquantized BF16 weights, BF16 activations and KV cache, TP=1, PCP=2, DCP=2,
`ag_rs`, interleave 1, block size 64, and eager execution without CUDA graphs.
Each result directory preserves the exact configuration, worker metadata,
prompt token IDs, environment, and experiment-time local patch.

## Ordinary CUDA-event timing

Step latency is the maximum rank for each iteration. The table summarizes
three measured iterations, so p50 describes the short run and is not a tail
estimate.

| Prefix | Scope | Mean (ms) | p50 (ms) | Stdev (ms) |
| ---: | --- | ---: | ---: | ---: |
| 20K | `self_attn` | 6.450741 | 6.493440 | 0.091706 |
| 20K | `full_layer` | 6.902773 | 6.943904 | 0.087572 |
| 40K | `self_attn` | 11.660256 | 11.706816 | 0.087635 |
| 40K | `full_layer` | 12.111051 | 12.161312 | 0.089421 |
| 80K | `self_attn` | 22.551584 | 22.390720 | 0.279401 |
| 80K | `full_layer` | 23.013504 | 22.852673 | 0.286000 |

The complete per-rank samples are in the 20K, 40K, and 80K
[`events/results.json`](prefix20k/events/results.json) artifacts under their
respective prefix directories.

### Scaling assessment

The labels use binary K: 20K, 40K, and 80K are 20,480, 40,960, and 81,920
tokens. The scaling is consistent with a prefix-dependent attention path plus
a fixed per-step component:

| Prefix | `self_attn` vs 20K | Increase from previous point | `full_layer - self_attn` | `self_attn / full_layer` |
| ---: | ---: | ---: | ---: | ---: |
| 20K | 1.0000x | - | 0.452032 ms | 93.451% |
| 40K | 1.8076x | 5.209514 ms | 0.450795 ms | 96.278% |
| 80K | 3.4960x | 10.891328 ms | 0.461920 ms | 97.993% |

A three-point least-squares fit, with `L` in Ki tokens, gives
`self_attn_ms = 1.00508 + 0.268910 * L` (`R^2 = 0.999882`) and
`full_layer_ms = 1.45155 + 0.269091 * L` (`R^2 = 0.999875`). The marginal
`self_attn` cost is 0.260476 ms/Ki token from 20K to 40K and
0.272283 ms/Ki token from 40K to 80K, a 4.53% slope increase. Three points do
not establish a general performance model, but the measured points show
neither constant cost nor pure quadratic growth.

The dominant non-NCCL, context-dominated kernel groups in the one-sample nsys
captures also scale with the prefix. Values below are the mean GPU residency
across the two ranks; the GEMM row combines the context `kv_b_proj` GEMM
chunks.

| Kernel group | 20K (ms) | 40K (ms) | 80K (ms) |
| --- | ---: | ---: | ---: |
| Context `kv_b_proj` GEMMs | 2.124555 | 4.200471 | 8.379487 |
| `ConcatMLAKKernel` | 1.196078 | 2.376956 | 4.737318 |
| FlashAttention SM90 prefill | 1.110638 | 2.178267 | 4.325656 |
| `cp_gather_cache` | 0.322224 | 0.893263 | 2.271275 |

The first three groups are close to linear. Cache gathering grows faster over
these points and the context is split into one, two, and three workspace chunks
at 20K, 40K, and 80K, respectively. That chunking and page-gather behavior is
consistent with most of the small marginal-slope increase. The approximately
0.45 ms gap between full-layer and attention timing is prefix-independent, so
its fraction is naturally amortized as the prefix grows.

## Nsight Systems attribution

This first table is the initial capture at each prefix. The headline is the
slower rank for each scope. `T` is the scope GPU wall span, `C` is the union of
context and suffix-cache communication, `A` is the union of non-NCCL kernels,
`O` is their overlap, and `E = C - O`.

| Prefix | Scope | Rank | T (ms) | C (ms) | A (ms) | O (ms) | E (ms) | Exposed comm |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20K | `self_attn` | 1 | 8.202044 | 2.655605 | 5.014857 | 0 | 2.655605 | 32.3774% |
| 20K | `full_layer` | 1 | 8.713305 | 2.655605 | 5.316103 | 0 | 2.655605 | 30.4776% |
| 40K | `self_attn` | 0 | 12.218952 | 0.876544 | 10.008775 | 0 | 0.876544 | 7.1736% |
| 40K | `full_layer` | 0 | 12.814441 | 0.876544 | 10.310248 | 0 | 0.876544 | 6.8403% |
| 80K | `self_attn` | 0 | 23.010458 | 1.870946 | 20.249911 | 0 | 1.870946 | 8.1309% |
| 80K | `full_layer` | 0 | 23.576251 | 1.870946 | 20.550264 | 0 | 1.870946 | 7.9357% |

The independent CUDA events recorded during the three nsys captures were
8.352256/8.800416 ms, 12.412128/12.940992 ms, and
23.175903/23.684959 ms for `self_attn`/`full_layer`, respectively.

Relative to the ordinary means, those nsys-run CUDA events are
29.48%/27.49%, 6.45%/6.85%, and 2.77%/2.92% longer. The decreasing relative
gap is consistent with a fixed profiler/launch perturbation being amortized,
but each point is only one independently captured sample.

### Repeated 20K captures

Three additional independent 20K captures tested whether the initial 32.38%
headline was caused by an invalid or uniquely unlucky run. This table reports
`self_attn`; the CUDA-event column is measured inside the same nsys run.

| Capture | Slow rank | T (ms) | A (ms) | C (ms) | Context (ms) | Suffix (ms) | C / T | CUDA event (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 1 | 8.202044 | 5.014857 | 2.655605 | 0.695581 | 1.960024 | 32.3774% | 8.352256 |
| Rerun 1 | 0 | 8.124435 | 5.036972 | 2.310181 | 0.787650 | 1.522531 | 28.4350% | 8.321440 |
| Rerun 2 | 1 | 8.130890 | 5.010406 | 2.322243 | 0.752481 | 1.569762 | 28.5607% | 8.343872 |
| Rerun 3 | 0 | 7.539083 | 5.037510 | 1.862915 | 0.699201 | 1.163714 | 24.7101% | 7.713344 |

Across all four captures, headline `C / T` has a 28.4979% median and a
24.7101%-32.3774% range. `C` has a 2.316212 ms median, while `A` stays within
5.010406-5.037510 ms. On the other rank, `C` stays within
0.129569-0.131488 ms and `C / T` within 1.8212%-1.9327%. The slow rank sequence
is 1, 0, 1, 0, so the imbalance follows process launch/progress phase rather
than a consistently slower device or partition.

The repeated result changes the confidence, not the mechanism. The initial
32.38% is the high end of a reproducible critical-rank residency range rather
than a bad 20K attention-compute sample. It still must not be interpreted as
pure link-transfer cost: almost all of the cross-rank difference is time for
which an NCCL kernel is resident while waiting for its peer. The four nsys-run
CUDA events have an 8.332656 ms median, 29.17% above the 6.450741 ms ordinary
mean, confirming that this short case is especially sensitive to profiler and
launch synchronization. The complete aggregate is in
[`prefix20k/nsys-reruns/summary.csv`](prefix20k/nsys-reruns/summary.csv).

### Why the communication ratios are not monotonic

The initial cross-prefix decomposition shows why these captures do not form a
communication scaling law. `Context` and `suffix` are NCCL GPU-kernel
residency, and bandwidth uses the physical context send volume. `C / T` equals
the reported exposed-communication ratio because no same-rank non-NCCL kernel
overlaps these collectives.

| Prefix | Rank | Context (ms) | Suffix (ms) | Physical context GB/s | C / T |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20K | 0 | 0.117696 | 0.013792 | 201.396 | 1.933% |
| 20K | 1 | 0.695581 | 1.960024 | 34.077 | 32.377% |
| 40K | 0 | 0.442816 | 0.433728 | 106.808 | 7.174% |
| 40K | 1 | 0.235102 | 0.013280 | 201.174 | 2.057% |
| 80K | 0 | 0.972577 | 0.898369 | 97.146 | 8.131% |
| 80K | 1 | 0.444959 | 0.013184 | 212.340 | 2.051% |

The suffix payload is exactly 147,456 bytes at every size. Its approximately
13 microsecond residency on the short-residency rank is stable, while the
initial slow rank records 1.960, 0.434, and 0.898 ms. In all four 20K captures,
the slow-rank suffix residency is 1.164-1.960 ms and the peer remains near
13 microseconds. Those larger values cannot be caused by prefix volume. An
NCCL kernel can remain resident while its rank waits for the peer process to
launch or make progress. Process scheduling and profiler perturbation also
lower apparent context bandwidth on the critical rank. The longer residency
alternates between ranks, another indication of launch/progress skew rather
than a backend-, device-, or length-dependent payload effect.

The timing trend is therefore reasonable, but the ratios require two distinct
interpretations:

- The ordinary 20K/40K/80K step times are internally consistent and nearly
  linear. Doubling is less than 2x because of the fitted fixed cost; it moves
  closer to 2x as that cost is amortized.
- The 40K and 80K headline communication residencies of 7.17% and 8.13% are
  mutually plausible one-sample critical-rank observations. Context compute
  and DCP context bytes both grow approximately as `O(L)`, so their ratio
  should approach a constant rather than double with prefix length. Extra
  chunks and sub-millisecond rank wait explain the small increase.
- The 20K critical-rank value is reproducibly high under this nsys setup:
  24.71%-32.38% over four captures. The fixed suffix wait is divided by a much
  shorter compute span, so it occupies a larger fraction than at 40K or 80K.
  The peer's stable approximately 1.9% and the alternating slow rank show that
  this is a synchronization-residency proportion, not a representative
  payload-transfer or production causal fraction.

## Communication payloads

The DCP context row is the physical implementation volume per rank. The unique
prefix baseline is `prefix_tokens * 576` bytes. PCP suffix-cache and hidden
restore payloads remain fixed across prefix sizes.

| Prefix | Unique baseline | DCP context send/recv | Amplification | PCP suffix send/recv | Hidden restore send/recv |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20K | 11,796,480 | 23,703,552 | 2.009375x | 147,456 | 1,835,008 |
| 40K | 23,592,960 | 47,296,512 | 2.0046875x | 147,456 | 1,835,008 |
| 80K | 47,185,920 | 94,482,432 | 2.00234375x | 147,456 | 1,835,008 |

Every measured iteration and both ranks passed the cache-hit, PCP partition,
collective count, tensor-shape, and byte-volume validators.

## Comparison with the 2026-08-08 automatic-backend run

The earlier 80K result used automatic selection and resolved to outer
`FLASH_ATTN_MLA`; this result explicitly resolves to `FLASHMLA`. Both runs,
however, set `use_pcp=true` and resolved the prefill backend to
`FlashAttnPrefillBackend`. In `MLACommonMetadataBuilder`, PCP makes
`split_decodes_and_prefills` preserve cached-prefix extensions as prefills.
The measured request therefore has 256 global MHA tokens, 128 on each PCP
rank, and zero MQA/decode tokens, so neither `FlashMLAImpl.forward_mqa` nor
`FlashAttnMLAImpl.forward_mqa` is called. Both runs execute the same common DCP
gather, `kv_b_proj`, concat, FlashAttention prefill, and merge path.

The trace supports that source-level explanation. At 80K, both captures have
the same key-kernel topology and counts; their non-NCCL kernel residencies are
nearly identical:

| Rank-0 kernel group | Earlier `FLASH_ATTN_MLA` (ms) | `FLASHMLA` outer (ms) | Change |
| --- | ---: | ---: | ---: |
| Context `kv_b_proj` GEMMs | 8.548263 | 8.397514 | -1.764% |
| `ConcatMLAKKernel` | 4.738420 | 4.738213 | -0.004% |
| FlashAttention SM90 prefill | 4.328627 | 4.323781 | -0.112% |
| `cp_gather_cache` | 2.284954 | 2.269859 | -0.661% |
| All non-NCCL kernels, union `A` | 20.417959 | 20.249911 | -0.823% |

The ordinary event means were likewise nearly unchanged, while the independent
one-sample nsys capture was modestly longer:

| Metric | Earlier auto backend | FlashMLA | Change |
| --- | ---: | ---: | ---: |
| CUDA events `self_attn` mean | 22.547552 ms | 22.551584 ms | +0.018% |
| CUDA events `full_layer` mean | 22.941664 ms | 23.013504 ms | +0.313% |
| nsys `self_attn` T | 22.673216 ms | 23.010458 ms | +1.487% |
| nsys `full_layer` T | 23.198591 ms | 23.576251 ms | +1.628% |
| nsys headline C | 1.562842 ms | 1.870946 ms | +19.714% |

Most of the nsys delta is the 0.308104 ms increase in headline NCCL residency,
not a slower attention kernel. The runs used different installed GPU drivers
(550.144.03 and 570.195.03), different process launches, and separate
one-sample nsys captures. The ordinary three-sample difference is only 0.018%
for attention. The defensible conclusion is that the two results agree within
short-run/profiler variability for this shared prefill path. They provide no
measurement of FlashMLA-versus-FlashAttention MLA decode performance; that
requires a decode/MQA workload in which the outer backend kernel actually
launches.

## Reports and integrity

Nsight Systems 2025.1.1 captured credential-like values from an ancestor
environment despite launching the target under a minimal environment. The raw
reports therefore remain local and are not in Git. The committed reports use
same-length replacements for those captured values, preserve file size and
trace structure, and were successfully exported and reanalyzed by nsys.

| Prefix | Sanitized report | SHA-256 |
| ---: | --- | --- |
| 20K | [`prefix20k_suffix256.sanitized.nsys-rep`](prefix20k/nsys/prefix20k_suffix256.sanitized.nsys-rep) | `e974d74db56d16636de84ed8dcf9575d998abaa0b75a5ded3e9d2a91d9f0367f` |
| 20K rerun 1 | [`prefix20k_suffix256.sanitized.nsys-rep`](prefix20k/nsys-reruns/rerun-1/prefix20k_suffix256.sanitized.nsys-rep) | `05c23ca60c5bf1d8688cc641a9d34ce0f634da22a3ab2d074548252cbe9a10b6` |
| 20K rerun 2 | [`prefix20k_suffix256.sanitized.nsys-rep`](prefix20k/nsys-reruns/rerun-2/prefix20k_suffix256.sanitized.nsys-rep) | `a17ec6ad881c28450f72abb3fbdc387936a91830e881347c432f24ee3a76f1ff` |
| 20K rerun 3 | [`prefix20k_suffix256.sanitized.nsys-rep`](prefix20k/nsys-reruns/rerun-3/prefix20k_suffix256.sanitized.nsys-rep) | `a3e9d61a4f9659fa0b6cab0c5201296872f6a67d9457c522caa8198e47da94d9` |
| 40K | [`prefix40k_suffix256.sanitized.nsys-rep`](prefix40k/nsys/prefix40k_suffix256.sanitized.nsys-rep) | `94b8add10047c8ae57945355b501416450e4070d8a39e79de4ede8bcb28bdabb` |
| 80K | [`prefix80k_suffix256.sanitized.nsys-rep`](prefix80k/nsys/prefix80k_suffix256.sanitized.nsys-rep) | `a799a60e91d931ac61d1d7de3788c6108a2eea688d52dfaa5aeed6e760886bc9` |

Each adjacent `redaction.json` records the raw and sanitized hashes, report
size, replacement method, and variable names. Each `precise-analysis/`
directory also contains an allowlisted portable SQLite, exact interval tables,
kernel records, and JSON/CSV summaries. The coarse overview CSV and raw GPU
trace CSVs are retained for independent inspection.

## Reproduction

Run the ordinary 20K and 40K points:

```bash
for prefix_tokens in 20480 40960; do
  PCP_DCP_MLA_PROFILE=1 VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
    .venv/bin/python \
    benchmarks/experimental/pcp_dcp_mla_prefix/run_experiment.py \
    --mode events --prefix-tokens "$prefix_tokens" \
    --allow-non-contract-shape \
    --output-dir "artifacts/pcp-dcp-flashmla-${prefix_tokens}-events"
done
```

Run the 80K contract point by omitting `--prefix-tokens` and
`--allow-non-contract-shape`. Capture nsys reports at all three sizes with:

```bash
for prefix_tokens in 20480 40960 81920; do
  PREFIX_TOKENS=$prefix_tokens MODE=nsys \
    RUN_DIR="artifacts/pcp-dcp-flashmla-${prefix_tokens}-nsys" \
    benchmarks/experimental/pcp_dcp_mla_prefix/run_nsys.sh
done
```

Repeat the attribution-sensitive 20K capture independently:

```bash
for repeat in 1 2 3; do
  PREFIX_TOKENS=20480 MODE=nsys RUN_OVERVIEW=0 \
    RUN_DIR="artifacts/pcp-dcp-flashmla-20k-rerun-${repeat}/nsys" \
    benchmarks/experimental/pcp_dcp_mla_prefix/run_nsys.sh
done
```

The driver defaults to `--attention-backend FLASHMLA` and verifies the worker
implementation. Reanalyze any committed report with `analyze_nsys.py`, the
matching `--unique-prefix-send-bytes`, and an installed nsys executable.

## Limitations

- This is attribution-oriented eager/NVTX execution, not production graph
  mode.
- Dummy BF16 weights and a deliberately dense layer 0 are not representative
  of an average DeepSeek-V3 MoE layer.
- Three ordinary repetitions characterize short-run stability, not tails.
- Each nsys capture contains one profiled iteration; 20K has four independent
  captures, while 40K and 80K each have one.
- NCCL timeline exposure does not prove equivalent end-to-end causal delay.
- The unsanitized reports cannot be committed safely on this node; manifests
  document every same-length credential redaction applied to the committed
  reports.
