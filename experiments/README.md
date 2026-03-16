# DP 负载不均衡 Baseline 实验

在 Data Parallel 场景下，多个长短不均的请求（1K, 2K, ... 64K tokens）会导致 DP 卡之间负载不均衡。本实验量化此问题，为后续 CP（Context Parallelism）优化提供对比基准。

## 文件说明

| 文件 | 说明 |
|------|------|
| `dp_imbalance_baseline.py` | 主实验脚本，启动多个 DP rank 进程并收集 metrics |
| `plot_dp_imbalance.py` | 可视化脚本，读取 JSON 结果生成 4 张图 |
| `run_baseline.sh` | 运行脚本，封装常用实验命令 |

## 实验原理

采用**离线 benchmark**（LLM 类），每个 DP rank 作为独立进程创建 LLM 实例，处理预分配的不同长度请求。通过 `RequestOutput.metrics`（`RequestStateStats`）获取 per-request 的时序数据。

输出长度保持一致（默认 128 tokens），以隔离 prefill 阶段的差异。

## 请求分布预设

以 `dp_size=2, lengths=[1K, 2K, 4K, 8K, 16K, 32K]` 为例：

| 预设 | Rank 0 | Rank 1 | 说明 |
|------|--------|--------|------|
| `extreme` | [1K, 2K, 4K] | [8K, 16K, 32K] | 短 vs 长，最大不均衡 |
| `interleaved` | [1K, 4K, 16K] | [2K, 8K, 32K] | 交错但总量不均 |
| `random` | 随机分配 | 随机分配 | 随机打乱分配 |
| `uniform` | 贪心装箱 | 贪心装箱 | 尽量让每个 rank 总 token 数接近 |
| `custom` | 用户通过 JSON 文件指定 | | 完全自定义 |

## 使用方法

### 快速验证（smoke test）

```bash
bash experiments/run_baseline.sh smoke
```

### 运行单个预设

```bash
python experiments/dp_imbalance_baseline.py \
    --model Qwen/Qwen2.5-7B \
    --dp-size 2 \
    --distribution extreme \
    --lengths 1024,2048,4096,8192,16384,32768 \
    --output-len 128 \
    --max-model-len 32768 \
    --output-dir results/extreme
```

### 运行全部预设

```bash
bash experiments/run_baseline.sh all
```

### 通过环境变量覆盖配置

```bash
MODEL=Qwen/Qwen2.5-0.5B DP_SIZE=2 MAX_MODEL_LEN=4096 \
    bash experiments/run_baseline.sh extreme
```

### 生成可视化

```bash
python experiments/plot_dp_imbalance.py --input-dir results/extreme
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen2.5-7B` | 模型名称或路径 |
| `--dp-size` | `2` | DP 并行度 |
| `--tp-size` | `1` | TP 并行度 |
| `--distribution` | `extreme` | 分布预设 |
| `--lengths` | `1024,2048,4096,8192,16384,32768` | 逗号分隔的 prompt 长度列表 |
| `--num-requests-per-length` | `2` | 每个长度的请求数 |
| `--output-len` | `128` | 每个请求的输出 token 数 |
| `--output-dir` | `results/baseline` | 结果输出目录 |
| `--max-model-len` | 自动 | 模型最大上下文长度 |
| `--gpu-memory-utilization` | `0.9` | GPU 显存利用率 |
| `--enforce-eager` | `false` | 禁用 CUDA graph 以节省显存 |
| `--seed` | `42` | 随机种子 |

## 输出格式

每个 rank 生成一个 `rank_N.json`，结构如下：

```json
{
  "dp_rank": 0,
  "model": "Qwen/Qwen2.5-7B",
  "distribution": "extreme",
  "total_time": 12.3,
  "total_prompt_tokens": 7168,
  "requests": [
    {
      "request_id": "rank0_req0",
      "prompt_len": 1024,
      "output_len": 128,
      "e2e_latency": 2.5,
      "prefill_time": 0.8,
      "decode_time": 1.7,
      "first_token_latency": 0.85
    }
  ]
}
```

## 可视化图表

| 图表 | 文件名 | 说明 |
|------|--------|------|
| Per-Rank Total Time | `1_total_time.png` | 每个 rank 的总完成时间（柱状图） |
| Per-Request E2E Latency | `2_e2e_latency.png` | 每个请求的 e2e 延迟，按 rank 着色（分组柱状图） |
| Prefill Time vs Prompt Length | `3_prefill_vs_length.png` | prefill 时间与输入长度的关系（散点图） |
| Timeline | `4_timeline.png` | 每个 rank 上请求的执行时间线（Gantt 图） |

## 注意事项

- 4090 (24GB) 上 Qwen2.5-7B 的 `max_model_len` 需根据实际显存调整（建议 ≤ 32K）
- 每个长度至少 2 个请求以减少噪声
- 建议先用小模型（如 Qwen2.5-0.5B）和短长度（512, 1K, 2K）跑通流程
