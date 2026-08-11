# FlashMLA sparse 8K priming 问题分析

## 结论

这不是 80K 序列本身应该如此慢，也不是 65,536 token 的模型长度限制。
问题出在实验为 TP=1、PCP=2、DCP=2 强制启用的 FlashMLA FP8
mixed-batch 路径：它用 sparse **decode** kernel 处理大块 prefill。8,192 个
global priming tokens 经 PCP 分片后，每个 rank 一次向该 kernel 提交约 4,096
个 query tokens；DCP 后 kernel 看到 128 个 query heads。FlashMLA 会为该形状
临时分配约 2 GiB 的 FP32 `o_accum`，并形成一个远超正常 decode 形状的 kernel
工作量。80K 首次运行在已计算 65,536 tokens、准备处理下一个 8,192-token
chunk 时失去进展；只把 priming chunk 改成 2,048 后即可越过该位置并在数秒内
完成。

因此，65,536 是问题暴露的位置，不是已经证明的边界条件。可由现有证据确认的
根因范围是“mixed-FP8 sparse decode 的大 query chunk”；究竟是临时内存压力、
allocator/工作区压力，还是该超常 kernel 形状内部的进度问题，没有单独的
kernel reproducer，不能进一步武断归因。按用户决定，本次不再做修改后的 80K
E2E 复测。

上游已经有 [vLLM PR #49357](https://github.com/vllm-project/vllm/pull/49357)
处理同一类问题，所以没有再创建重复 PR。本分支把它叠加到
[vLLM PR #46514](https://github.com/vllm-project/vllm/pull/46514)，并保留后者
所需的 DCP LSE 语义。

## 为什么 8K chunk 会出问题

FlashMLA 的 `sparse_decode_fwd` 在每次调用中分配：

```text
o_accum: [batch + num_sm_parts, s_q, h_q, d_v], float32
```

源码见
[`sparse_decode.h`](https://github.com/deepseek-ai/FlashMLA/blob/a8f794d1251cbfd88a5011445dd5582289c727e4/csrc/api/sparse_decode.h#L463-L466)。
H100 上本实验的大 query 形状有 `batch=1`、`num_sm_parts=1`、`h_q=128`、
`d_v=512`。不同 global priming chunk 对应的主 scratch 为：

| Global chunk | PCP rank-local `s_q` | `o_accum` |
| ---: | ---: | ---: |
| 8,192 | 4,096 | 2 GiB |
| 2,048 | 1,024 | 512 MiB |

8K 形状还同时需要约 576 MiB 的 BF16 query、512 MiB 的 BF16 attention output、
top-k index/score buffers 和其他工作区。它虽然只有一个 attention 层，却不代表
一次 attention 调用很小；这里的临时张量由 `Q × heads × 512` 决定。GPU
100% utilization 但带宽和功耗很低也不符合“只是在做正常的长序列计算”，而更
像某个超大 decode-style launch 没有正常推进。

2K chunk 能越过相同的 65K 位置，排除了两个解释：

- 不是模型在 65,536 token 处有硬长度限制；
- 不是单层 attention 的正常长上下文计算就需要二十分钟。

它也说明 2K 是有效的实验 workaround，但在修改后 E2E 重跑前，不能把 subchunk
修复写成已验证解决了这次 stall。

## #49357 如何修复

PR #49357 增加 `VLLM_FLASHMLA_SPARSE_MAX_SCRATCH_MB`，默认 512 MiB。metadata
builder 不再把 mixed batch 的所有 query tokens 作为一次 `s_q` 传给 decode
kernel，而是为多个 query subchunks 分别生成 scheduler metadata；forward
逐块调用 kernel，并把每块输出写回原 token slice。

按 query 维分块在数学上成立：每个 query row 有自己的 top-k keys 和 softmax，
不同 query rows 之间没有 reduction。只要保持 token 顺序并逐块拼接 output/LSE，
结果应与一次大调用一致。

原 #49357 的 chunk 上限固定按 64 padded heads 计算：

```text
512 MiB / (2 splits × 64 heads × 512 values × 4 bytes) = 2,048 tokens
```

这能把本实验每 rank 的 4,096-token 调用拆成两个调用，但 DCP 后实际 kernel
形状是 128 heads，所以每个 subchunk 的 `o_accum` 仍是 1 GiB，而非配置所表达
的 512 MiB。本分支进一步按实际 `fp8_decode_padded_heads` 计算上限：H100 默认
预算下 64 heads 为 2,048 tokens，128 heads 为 1,024 tokens。8K global chunk
因而预计在每个 rank 上拆成四个 1,024-token kernel，每次主 scratch 为 512 MiB。

## 与 #46514 叠加时需要补的语义

PR #49357 早于 #46514，不能原样 cherry-pick 后直接认为 DCP 正确：

- #46514 要求 mixed-FP8 forward 返回 `(output, lse)`，供跨 DCP rank 的 LSE
  merge 使用；原 #49357 的 loop 只拼接 output，并丢弃每个 subchunk 的 LSE。
- 某个 DCP rank 对一行没有本地 top-k candidate 时，FlashMLA kernel 的 output/LSE
  未定义。#46514 将该行显式改成 `(0, -inf)`，这是跨 rank LSE merge 的恒等元；
  subchunk 后仍必须保留该处理，否则 NaN 可在 `0 * NaN` 中存活。
- head padding 和 scratch 上限必须按 kernel 实际看到的 heads，而不是固定 64
  heads 计算。

本分支的整合实现逐块保存转置后的 LSE，全部拼接后再做 empty-row
neutralization；同时增加覆盖 64/128-head budget、slice 完整性，以及跨两个
subchunks 的 DCP empty-row 行为测试。按照用户决定，这些测试未完成最终 pytest
执行；仅完成了针对改动文件的 `ruff-check`。

## 为什么 dense 没有这个问题

此前所谓 dense `FLASHMLA` 三点实验中，outer backend 虽然是 `FLASHMLA`，但
PCP 会把带 cached prefix 的 256-token extension 保留在 MHA prefill 路径，实际
调用的是公共 `FlashAttnPrefillBackend`。trace 中也是 FlashAttention SM90 prefill
kernels，而不是 FlashMLA dense decode kernel。

这条 dense prefill 路径按 K/V tile 流式计算，并对长 context 做 workspace
chunking；它不会分配 sparse decode 的
`[splits, Q, heads, 512]` FP32 `o_accum`。所以 dense 可能计算更多、耗时更长，
但不会碰到相同的 8K mixed-decode scratch cliff。BF16 sparse 的独立 prefill
路径同样使用专门的 `flash_mla_sparse_fwd`，也不是这个 FP8 decode 路径。

## 20K / 40K / 80K 耗时是否合理

所有正式结果都是 batch size 1：每次只有一个 user request，`max_num_seqs=1`。
PCP metadata 中看到的两个条目是 virtual CP chunks，不是两个 requests。每次
测量固定计算 256 个 global suffix query tokens（每 rank 128），只改变 cached
prefix 长度。

| Prefix | Dense common prefill `self_attn` | Sparse `self_attn` | Dense / sparse |
| ---: | ---: | ---: | ---: |
| 20K | 6.451 ms | 3.643 ms | 1.77x |
| 40K | 11.660 ms | 3.916 ms | 2.98x |
| 80K | 22.552 ms | 4.130 ms | 5.46x |

`self_attn` 只有在完整 prefill 且 `Q=K=L` 时才是二次复杂度。本次测的是固定
`Q=256` 的 cached-prefix extension：

- dense attention 是 `O(Q × L)`，Q 固定，所以对 prefix 长度近似线性；
- sparse indexer 是 `O(Q × L)`，同样只有较小的线性项；
- sparse attention 是 `O(Q × topk)`，top-k 固定为 2,048，近似常数；
- suffix communication 的 tensor shapes 固定，不随 prefix 长度增长。

所以 dense 从 20K 到 80K 增长 3.50x、而不是 16x，是“线性长度项 + 固定成本”
的合理结果；sparse 只增长 13.4%，也符合“较大的固定成本 + 较小的线性 indexer
成本”。如果测量从空 cache 构建完整 L-token prefix 的累计时间，随着每个 chunk
面对越来越长的 K，累计工作才可能接近二次增长；那不是上表的 suffix latency。

## Indexer 在哪个 NVTX scope

Indexer 计算在 `self_attn` 内部的 `sparse_indexer_scope`，该 scope 包住
`torch.ops.vllm.sparse_attn_indexer`。其中还可能嵌套：

- `suffix_cache_comm` 且 `tensor=indexer_k`：新 suffix index keys 的 PCP
  all-gather；
- `sparse_indexer_comm`：各 DCP rank 的 local top-k score/index candidates
  all-gather，再做 global top-k；
- 后续 `sparse_attention_compute` 才是 FlashMLA FP8 sparse attention kernel，
  不属于 indexer。

稳定 rank 的 indexer 非 NCCL kernel 分布如下；括号为占 indexer local compute
union 的比例：

| Prefix | Score cached keys | Local top-k | Pack | Global merge | Other | Total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20K | 0.040 ms (15.1%) | 0.201 (75.1%) | 0.003 | 0.010 | 0.013 | 0.268 ms |
| 40K | 0.075 ms (24.2%) | 0.205 (66.1%) | 0.004 | 0.011 | 0.016 | 0.310 ms |
| 80K | 0.146 ms (35.8%) | 0.229 (56.1%) | 0.004 | 0.011 | 0.018 | 0.409 ms |

增长来自 score cached keys；local top-k、candidate packing 和 global merge 基本
固定。两 rank 的 local compute 总量分别约为 0.268/0.310/0.409 ms 和
0.271/0.306/0.404 ms，互相吻合。

## 通信占比怎么理解

单次 Nsight slow-rank 的 sparse NCCL residency 占比为 50%/82%/61%，不能当作
20K/40K/80K 的通信 scaling law。三点的通信 payload 完全相同，而 slow rank
中很小的 collective 可驻留 0.4--6 ms，说明主要计入了等待 peer 到达/推进的
时间以及 profiler 扰动。稳定 rank 的 NCCL residency 都约为 0.13 ms，`C/T`
均约 2.8%。因此：

- CUDA-event 的普通运行延迟趋势可以用于长度比较；
- Nsight 的 `C/T` 是“collective GPU kernel residency，包括等待”，不是纯网络
  transfer time，也不是可直接消除的因果通信比例；
- 40K 的 82% 是最明显的 rank-arrival skew，不是 40K payload 特别大。

完整原始结果、indexer 两 rank 分布和经过清理的 Nsight 报告已在提交
[`3ef23c25f`](https://github.com/Livinfly/vllm/commit/3ef23c25f6105d913ff0f6226255ed15d06db53d)
中保存。

## 验证状态与限制

- 8K 失败和 2K workaround 来自同一套单请求、单层 dummy
  DeepSeek-V3.2-Exp、2x H100 实验。
- #49357 与 #46514 的原始版本分别存在并可审查；本分支是两者的本地集成分析，
  不会创建 PR。
- 已执行 duplicate-work 检查，#49357 正在解决 issue #44545 的相同根因。
- 已通过改动文件的 `ruff-check`；按用户要求停止 pytest 和修改后 E2E 重跑，
  因此不能声称 80K/8K 已被本分支实测修复，也没有模型精度评估。
- #46514 本身仍会拒绝本实验的 TP1、PCP2、DCP2、128-head shape；先前性能实验
  使用了显式 mixed-FP8 opt-in。这个 opt-in 不包含在本修复分支中。
