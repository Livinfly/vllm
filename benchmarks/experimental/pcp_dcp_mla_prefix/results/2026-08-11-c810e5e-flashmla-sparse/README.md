# FlashMLA sparse DeepSeek-V3.2-Exp 1L PCP2+DCP2 results

## Outcome

The batch-size-one sparse experiment completed on 2026-08-11 UTC in the
requested order: 80K, 40K, then 20K. Each point primes one cached prefix, runs
eight uncaptured warmups, and measures five distinct 256-token suffixes with
CUDA events. A separate process captures one suffix with Nsight Systems after
the same eight warmups.

Every measured call contains exactly one user request and `max_num_seqs=1`.
The two entries reported by PCP metadata are virtual context-parallel chunks,
not two user requests. The measured request has 256 computed prompt tokens,
128 local query tokens on each rank, and an exact 20,480/40,960/81,920-token
cache hit.

The worker resolves to `FLASHMLA_SPARSE` and `FlashMLASparseImpl`. The measured
suffix launches the `flashmla_fp8_mixed_batch` sparse kernel with 128 query
heads and top-k 2048. The `FLASH_ATTN` prefill-backend log describes initial
cache priming; it is not the kernel used by the captured cached-prefix suffix.

This proves that the target TP=1, PCP=2, DCP=2 `ag_rs` suffix workload can run
on the local PR [#46514](https://github.com/vllm-project/vllm/pull/46514)
overlay with the explicit mixed-FP8 opt-in added for this experiment. It does
not show that the PR alone accepts this configuration: its existing head-count
guard still rejects the DeepSeek-V3.2-Exp TP=1 shape without the opt-in.

## Workload and environment

| Item | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V3.2-Exp` |
| Model shape | One layer-0 decoder block, dummy weights |
| Attention shape | 128 heads, top-k 2048, index head dim 128 |
| Weight dtypes | 2,467,585,024 BF16 parameters; 256 FP32 indexer-norm parameters |
| Activation / KV cache | BF16 / `fp8_ds_mla` |
| Parallelism | TP=1, PCP=2, DCP=2, `ag_rs`, interleave 1 |
| Request batch | One user request, `max_num_seqs=1` |
| Measured suffix | 256 global tokens, 128 local tokens per rank |
| GPUs | 2x NVIDIA H100 80GB HBM3 |
| Driver / PyTorch CUDA | 570.195.03 / 12.9 |
| PyTorch / NCCL | 2.13.0 / 2.29.7 |
| Nsight Systems | 2025.1.1.0 |
| Execution | Eager, no CUDA graphs |
| Precompiled source SHA | `c810e5ee9976ad86b81d1277b53e76d0ee639414` |
| Experiment HEAD | `757c95ce262172fc360d5c65d43b7186a327df70` plus recorded patch |
| PR #46514 overlay | `757c95ce262172fc360d5c65d43b7186a327df70` |

The model is one complete decoder block, so cache priming and `full_layer`
also include its dense MLP. The primary `self_attn` scope isolates attention.
This matches the earlier one-layer baseline rather than constructing a custom
attention-only module.

### Cache-priming chunk issue

An 8,192-token priming chunk is pathological on the forced mixed-FP8 sparse
path. The first 80K attempt timed out after 300 seconds with the scheduler at
65,536 computed tokens and an 8,192-token next chunk. Raising the RPC timeout
did not make it valid: both workers remained in a low-power GPU/CPU spin for
about 20 minutes before the run was stopped.

Changing only `max_num_batched_tokens` for cache priming to 2,048 made the 80K
prefix cross 65,536 and finish normally in seconds. Thus the failure is not a
65,536-token model limit and not expected long-sequence complexity. It is a
large-query-shape problem in the decode-style mixed-FP8 priming path. All
formal points use the same 2,048-token priming chunk. This setting does not
change the measured 256-token suffix, its cache hit, or its kernel shape.

## Ordinary CUDA-event timing

Each iteration is the maximum of the two rank-local CUDA-event durations.
Five measured suffixes follow eight warmups.

| Prefix | Scope | Mean (ms) | p50 (ms) | p90 (ms) | Stdev (ms) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20K | `self_attn` | 3.643053 | 3.601280 | 3.907616 | 0.245799 |
| 20K | `full_layer` | 4.115738 | 4.053568 | 4.364729 | 0.228242 |
| 40K | `self_attn` | 3.916083 | 3.810816 | 4.372909 | 0.425663 |
| 40K | `full_layer` | 4.416704 | 4.344256 | 4.849696 | 0.410061 |
| 80K | `self_attn` | 4.130074 | 3.960832 | 4.651328 | 0.532515 |
| 80K | `full_layer` | 4.587872 | 4.428320 | 5.107059 | 0.531371 |

The complete per-rank samples and all range records are in each
[`events/results.json`](prefix20k/events/results.json) artifact.

### Scaling

| Prefix | `self_attn` vs 20K | Increase from prior point | `full_layer - self_attn` |
| ---: | ---: | ---: | ---: |
| 20K | 1.0000x | - | 0.472685 ms |
| 40K | 1.0749x | 0.273030 ms | 0.500621 ms |
| 80K | 1.1337x | 0.213990 ms | 0.457798 ms |

A three-point fit with `L` in Ki tokens is
`self_attn_ms = 3.53606 + 0.007722 * L` (`R^2 = 0.9339`). The fit is only a
description of these three short runs, but it captures the large fixed cost
and small prefix-dependent term. The p50 grows only 9.98% from 20K to 80K.

The sparse-attention kernel's ordinary event mean is nearly fixed at
0.311/0.328/0.318 ms for 20K/40K/80K. The Nsight `sparse_indexer_scope`
isolates the length-dependent work. On rank 0 its non-NCCL kernel union is
0.268/0.310/0.409 ms; a linear fit has `R^2 = 0.9984`. Rank 1 gives
0.271/0.306/0.404 ms and `R^2 = 0.9947`.

### Why `self_attn` is not quadratic here

A full prefill with `Q = K = L` has quadratic attention work. That is not the
measured operation. These runs hold the suffix query count at 256 globally
(128 per rank) while only the cached key length changes:

- the DSA indexer scores fixed `Q` against `L`, so its variable work is
  `O(Q * L) = O(L)`;
- sparse attention consumes fixed `Q * topk`, with top-k fixed at 2048, so it
  is approximately constant with prefix length;
- all measured communication tensors depend on fixed Q, heads, and top-k, so
  their byte counts are constant;
- projections, launch overhead, and the rest of the one-layer suffix are also
  mostly fixed.

The total is therefore a large fixed component plus a small linear indexer
component. A four-times-longer prefix should not make this cached-prefix
`self_attn` sixteen times slower. The one-time operation that builds the full
prefix cache is a different workload and can have near-quadratic aggregate
prefill work.

## Comparison with dense `FLASHMLA` / `FLASH_ATTN_MLA`

The earlier dense three-length result explicitly requested outer `FLASHMLA`,
but PCP kept the cached-prefix extension on the common
`FlashAttnPrefillBackend` dense-MHA path. The earlier automatic 80K result
resolved to outer `FLASH_ATTN_MLA` and used the same common path. Their 80K
ordinary means agree within 0.02%, so the dense three-length sweep is the
appropriate scaling baseline for both outer names.

| Prefix | Dense common prefill (ms) | Sparse (ms) | Dense / sparse | Sparse reduction |
| ---: | ---: | ---: | ---: | ---: |
| 20K | 6.450741 | 3.643053 | 1.7707x | 43.53% |
| 40K | 11.660256 | 3.916083 | 2.9775x | 66.42% |
| 80K | 22.551584 | 4.130074 | 5.4603x | 81.69% |

The automatic `FLASH_ATTN_MLA` 80K mean was 22.547552 ms, giving the same
5.4594x comparison. This is not a pure FlashMLA-versus-FlashAttention kernel
comparison:

- dense executes attention over the full cached context and gathers context
  state whose volume grows with `L`;
- sparse scans a low-dimensional index, retains only 2048 positions, and runs
  the actual FlashMLA FP8 sparse kernel on those positions;
- dense context communication grows from 23.7 to 94.5 MB per rank over these
  points, while every sparse suffix communication payload is fixed.

The widening speedup is therefore the expected algorithm/path difference.
The dense baseline analysis is in
[`../2026-08-11-c810e5e-flashmla/README.md`](../2026-08-11-c810e5e-flashmla/README.md),
and the automatic outer-backend result is in
[`../2026-08-08-c810e5e/README.md`](../2026-08-08-c810e5e/README.md).

## Nsight Systems attribution

`T` is the GPU wall span of the scope on the rank with the larger span. `C` is
the union of classified NCCL kernels, `A` is the union of non-NCCL kernels,
`O` is their overlap, and `E = C - O`. There is no same-rank kernel overlap in
these captures.

| Prefix | Rank | T (ms) | A (ms) | C (ms) | C / T |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20K | 1 | 5.392296 | 0.769153 | 2.721412 | 50.47% |
| 40K | 1 | 9.701551 | 0.809505 | 7.933005 | 81.77% |
| 80K | 1 | 6.611212 | 0.907393 | 4.034888 | 61.03% |

Those headline ratios are valid NCCL GPU-residency proportions for the single
captures, but they are not pure network-transfer fractions. The same fixed
payloads cannot physically take 2.72, 7.93, and 4.03 ms as a function of
prefix length. Rank 0 in the same traces is stable:

| Prefix | Rank-0 T (ms) | Rank-0 C (ms) | Rank-0 C / T |
| ---: | ---: | ---: | ---: |
| 20K | 4.656235 | 0.130785 | 2.81% |
| 40K | 4.781033 | 0.133857 | 2.80% |
| 80K | 4.749833 | 0.132768 | 2.80% |

The long-residency rank enters collectives earlier and remains resident while
waiting for its peer process to launch or progress. Nsight perturbation makes
that phase skew much larger than in ordinary execution. The CUDA events in the
three nsys processes are 5.626/9.876/6.823 ms, respectively, versus ordinary
means of 3.643/3.916/4.130 ms. Consequently, the 50%/82%/61% sequence is not a
communication scaling law.

### Slow-rank communication categories

| Prefix | Suffix-cache PCP AG (ms) | Indexer DCP AG (ms) | LSE DCP AG (ms) | Output DCP AR (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 20K | 1.347266 | 0.498977 | 0.488449 | 0.386720 |
| 40K | 6.153994 | 0.494817 | 0.872705 | 0.411489 |
| 80K | 2.382596 | 0.396097 | 0.862146 | 0.394049 |

The most variable category is three tiny suffix-cache all-gathers totaling
only 180,224 bytes. In the 40K trace, the 32 KiB indexer-K all-gather alone is
resident for 5.488 ms. That cannot be a bandwidth effect and identifies peer
arrival wait as the dominant source of the headline variation.

## Indexer NVTX scope

Indexer computation is now visible under `sparse_indexer_scope`, nested inside
`self_attn`. It wraps `torch.ops.vllm.sparse_attn_indexer`. Two communication
ranges can be nested within it:

- `suffix_cache_comm` with `tensor=indexer_k` is the PCP all-gather of the new
  suffix's index keys;
- `sparse_indexer_comm` is the DCP all-gather of local top-k score/index
  candidates before global top-k selection.

The analyzer's `A` for `sparse_indexer_scope` is the union of the remaining
non-NCCL indexer kernels, including scoring, local top-k, packing, and global
candidate merge. `sparse_attention_compute` begins after the indexer and wraps
only the FlashMLA FP8 sparse attention kernel.

| Prefix | Indexer T (ms) | Local kernel A (ms) | Nested C (ms) | C / T |
| ---: | ---: | ---: | ---: | ---: |
| 20K | 1.527395 | 0.270753 | 1.228066 | 80.40% |
| 40K | 6.320842 | 0.306368 | 5.983274 | 94.66% |
| 80K | 2.610341 | 0.404289 | 2.174948 | 83.32% |

The local kernel column is the useful length trend. The communication column
contains the same rank-wait artifact as the outer scope. On rank 0, indexer
communication is only 0.0347/0.0350/0.0349 ms at 20/40/80K.

### Indexer distribution by length and rank

The following table records both ranks rather than only the slow-rank
headline. `Gap` is `T - A - C + O`: GPU time between the first and last
indexer kernels that belongs to neither a compute kernel nor a classified NCCL
kernel. `Indexer/self` compares the indexer GPU span with the enclosing
`self_attn` GPU span on the same rank.

| Prefix | Rank | T (ms) | A (ms) | C (ms) | Gap (ms) | A / T | C / T | Gap / T | Indexer / self |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20K | 0 | 0.918467 | 0.267553 | 0.034688 | 0.616226 | 29.13% | 3.78% | 67.09% | 19.73% |
| 20K | 1 | 1.527395 | 0.270753 | 1.228066 | 0.028576 | 17.73% | 80.40% | 1.87% | 28.33% |
| 40K | 0 | 0.961313 | 0.309569 | 0.035008 | 0.616736 | 32.20% | 3.64% | 64.16% | 20.11% |
| 40K | 1 | 6.320842 | 0.306368 | 5.983274 | 0.031200 | 4.85% | 94.66% | 0.49% | 65.15% |
| 80K | 0 | 0.935746 | 0.408769 | 0.034912 | 0.492065 | 43.68% | 3.73% | 52.59% | 19.70% |
| 80K | 1 | 2.610341 | 0.404289 | 2.174948 | 0.031104 | 15.49% | 83.32% | 1.19% | 39.48% |

Rank 0 is the stable peer: its indexer is consistently about 20% of the
enclosing GPU span and its communication stays below 0.036 ms. Rank 1's local
compute agrees with rank 0, while its `T` and `C` vary together. This is the
same early-arrival wait seen at the outer attention scope, especially at 40K.
The large rank-0 gap is launch/control spacing inside the indexer span, not
unclassified GPU work; all NCCL kernels inside the scope pass fail-closed
classification.

The stable-rank local compute union `A` breaks down as follows. Parentheses are
the percentage of `A`, not of the full indexer span.

| Prefix | Score cached keys | Local top-k | Pack candidates | Global merge | Cache and other | Total A |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20K | 0.040416 (15.11%) | 0.200929 (75.10%) | 0.003168 (1.18%) | 0.009728 (3.64%) | 0.013312 (4.98%) | 0.267553 ms |
| 40K | 0.074913 (24.20%) | 0.204544 (66.07%) | 0.003616 (1.17%) | 0.010688 (3.45%) | 0.015808 (5.11%) | 0.309569 ms |
| 80K | 0.146336 (35.80%) | 0.229473 (56.14%) | 0.003968 (0.97%) | 0.010944 (2.68%) | 0.018048 (4.42%) | 0.408769 ms |

The score kernel is the clear length-dependent component: it grows from
0.040 to 0.146 ms as the prefix grows fourfold. Local top-k and the fixed-size
DCP candidate merge remain nearly flat, so their percentage falls. Rank 1 has
the same compute distribution within measurement noise: total `A` is
0.271/0.306/0.404 ms versus rank 0's 0.268/0.310/0.409 ms.

The nested communication distribution further exposes the wait asymmetry:

| Prefix | Rank | Indexer-K PCP all-gather | Candidate DCP all-gather |
| ---: | ---: | ---: | ---: |
| 20K | 0 | 0.006464 ms | 0.028224 ms |
| 20K | 1 | 0.729089 ms | 0.498977 ms |
| 40K | 0 | 0.006656 ms | 0.028352 ms |
| 40K | 1 | 5.488457 ms | 0.494817 ms |
| 80K | 0 | 0.006688 ms | 0.028224 ms |
| 80K | 1 | 1.778851 ms | 0.396097 ms |

These are component distributions from one Nsight capture per length, not
percentiles across the five ordinary event samples. The underlying per-kernel
records are retained in each `precise-analysis/kernels.csv`.

## Communication payloads

All values are per rank and are identical at 20K, 40K, and 80K.

| Category | Collective | Send bytes | Receive bytes |
| --- | --- | ---: | ---: |
| Indexer-K suffix cache | PCP all-gather | 32,768 | 32,768 |
| MLA latent suffix cache | PCP all-gather | 131,072 | 131,072 |
| RoPE suffix cache | PCP all-gather | 16,384 | 16,384 |
| Indexer top-k candidates | DCP all-gather | 2,097,152 | 2,097,152 |
| Attention LSE | DCP all-gather | 65,536 | 65,536 |
| Attention output | DCP all-reduce | 16,777,216 | 16,777,216 |
| Hidden-state restore | PCP all-gather | 1,835,008 | 1,835,008 |

Every warmup, ordinary measurement, and nsys measurement passed collective
count, byte volume, cache-hit, local-query, kernel-name, head-count, top-k, and
single-user-request validation.

## Reports and integrity

Raw nsys reports remain local because Nsight captured credential-like ancestor
environment values. The committed reports replace those values with
same-length asterisks. All three sanitized reports were independently exported
and reanalyzed; their headline and indexer-subscope summaries matched exactly.
All portable SQLite files pass `PRAGMA integrity_check`.

| Prefix | Sanitized report | SHA-256 |
| ---: | --- | --- |
| 20K | [`sparse_prefix20k_suffix256.sanitized.nsys-rep`](prefix20k/nsys/sparse_prefix20k_suffix256.sanitized.nsys-rep) | `b4d77b70fd1add782c9ffdff097641774cebdf32f7ec90b6018cf47c69062642` |
| 40K | [`sparse_prefix40k_suffix256.sanitized.nsys-rep`](prefix40k/nsys/sparse_prefix40k_suffix256.sanitized.nsys-rep) | `e78dd92b223b98d6097873cd7ddbe42b463c2ed920c5ef4d9218d23fa6e107a8` |
| 80K | [`sparse_prefix80k_suffix256.sanitized.nsys-rep`](prefix80k/nsys/sparse_prefix80k_suffix256.sanitized.nsys-rep) | `edcda1aa9525cfc38af107ad8d97608f6ddf0b658b70c2a2a149844c5cf9f434` |

Each adjacent redaction manifest records the source/sanitized hashes, report
size, replacement count, and redacted variable names. Each
`precise-analysis/` directory contains interval tables, kernel records,
JSON/CSV summaries, and an allowlisted portable SQLite trace.

## Reproduction

Run the ordinary points:

```bash
for prefix_tokens in 81920 40960 20480; do
  extra=()
  if (( prefix_tokens != 81920 )); then
    extra+=(--allow-non-contract-shape)
  fi
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  PCP_DCP_MLA_PROFILE=1 VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
    .venv/bin/python \
    benchmarks/experimental/pcp_dcp_mla_prefix/run_experiment.py \
    --workload sparse --mode events --prefix-tokens "$prefix_tokens" \
    --prefill-chunk-tokens 2048 --warmup-iterations 8 \
    --measured-iterations 5 "${extra[@]}" \
    --output-dir "artifacts/sparse-${prefix_tokens}-events"
done
```

Capture nsys at the same points:

```bash
for prefix_tokens in 81920 40960 20480; do
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 WORKLOAD=sparse MODE=nsys \
    PREFIX_TOKENS=$prefix_tokens \
    RUN_DIR="artifacts/sparse-${prefix_tokens}-nsys" \
    benchmarks/experimental/pcp_dcp_mla_prefix/run_nsys.sh \
    --prefill-chunk-tokens 2048 --warmup-iterations 8 \
    --measured-iterations 1
done
```

## Limitations

- PR #46514 is still an open local overlay, and TP1 PCP2 DCP2 requires the
  explicit experiment opt-in.
- The 8K mixed-FP8 priming shape stalls; 2K is a measured-workload-neutral
  workaround, not a kernel fix.
- Dummy weights do not permit an accuracy/model evaluation. This experiment
  validates execution shape and performance attribution only.
- Eager/NVTX execution is attribution-oriented, not production graph mode.
- Five ordinary samples characterize steady short-run behavior, not tails.
- Each nsys point is one capture. Its NCCL residency includes peer wait and
  must not be interpreted as pure transfer time or causal speedup potential.
