# AI Infra 101

> **从 Transformer 内部机制到生产级 LLM Systems：一套面向 AI Infra 学习与面试的压缩式 Handbook。**

这个仓库面向已经具备基本 Python / C++ / CUDA 能力，希望系统补齐 **LLM Inference / GPU Performance / Distributed Parallelism / Serving / Runtime / Profiling** 的学生和工程师。

它不是一门几十小时的完整课程，也不是一个堆满链接的 awesome-list。每个章节都按照下面的逻辑组织：

> **核心机制 → 公式/Shape → Worked Example → 常见误区 → 面试问题 → 最短延伸资料**

英文 README 是项目的主要对外入口；中文版本更适合作为快速导航。

---

## 知识栈

| Layer | 核心内容 | 入口 |
|---|---|---|
| 01 Transformer Execution | RMSNorm、RoPE、MHA/MQA/GQA、SwiGLU、MoE、Shape/GEMM | [阅读](docs/01-transformer-execution.md) |
| 02 LLM Inference | Prefill/Decode、KV Cache、Batching、Quantization、Speculative Decoding | [阅读](docs/02-llm-inference.md) |
| 03 GPU Performance | SIMT、Memory Hierarchy、Roofline、Fusion、FlashAttention | [阅读](docs/03-gpu-performance.md) |
| 04 Distributed Parallelism | Collective、TP/PP/DP/EP、Topology、Overlap | [阅读](docs/04-distributed-parallelism.md) |
| 05 LLM Serving | TTFT/TPOT、Continuous Batching、PagedAttention、Chunked Prefill、P/D | [阅读](docs/05-llm-serving.md) |
| 06 Runtime Internals | Scheduler → KV Manager → ModelRunner → Attention Backend | [阅读](docs/06-runtime-internals.md) |
| 07 Profiling | nsys/ncu、瓶颈分类、实验设计、Regression | [阅读](docs/07-profiling.md) |
| 08 Interview Prep | 手撕、推导、项目 Deep Dive、System Design | [阅读](docs/08-interview-prep.md) |

---

## “掌握”一个知识点到底意味着什么？

不要把“看过”当成“会了”。

- **Explain**：脱离资料，用 1–3 分钟准确解释机制。
- **Derive**：现场推 Shape / FLOPs / Memory / Bandwidth / Communication。
- **Implement**：写出核心伪代码或 PyTorch/CUDA-like 实现。
- **Diagnose**：面对真实性能问题，能够提出证据、profiling 路径和 trade-off。

对于 AI Infra 实习，我建议达到：

```text
整个 Stack          → Explain + Derive
Transformer/Tensor  → Implement
Inference/GPU       → Diagnose
```

---

## 没有整块时间的话怎么学？

直接按照：

```text
Transformer Execution
        ↓
Inference + KV 计算
        ↓
GPU / Roofline
        ↓
TP / PP / EP 推导
        ↓
Serving
        ↓
追一遍 vLLM/SGLang Request Path
        ↓
Profile 一个真实 workload
        ↓
Mock Interview
```

每一章现在都已经扩展成可以独立阅读的短教程，不需要先完整看一门公开课。

---

## 配套面试材料

- [100+ 道 AI Infra 自测题](interview/100-questions.md)
- [30 道核心题的一分钟答案](interview/core-answers.md)
- [手撕题 Checklist](interview/handwrite-checklist.md)
- [公式速查](cheatsheets/formulas.md)
- [Collective / Parallelism 速查](cheatsheets/collectives.md)
- [完整资料索引](resources/reading-list.md)

一分钟答案不是为了背诵，而是帮助你建立：

```text
Definition
→ Problem
→ Mechanism
→ Trade-off
→ Systems consequence
```

这种适合面试表达的结构。

---

## 可运行的小工具

仓库增加了几个零依赖 calculator，用来检查自己的手算结果。

### KV Cache

```bash
python tools/kv_cache_calculator.py \
  --layers 32 --kv-heads 8 --head-dim 128 \
  --seq-len 16384 --bytes-per-element 2 \
  --concurrency 16 --kv-budget-gib 32
```

### Roofline

```bash
python tools/roofline_estimator.py \
  --flops 2e12 --bytes 1e11 \
  --peak-tflops 100 --bandwidth-gbs 2000
```

### TP Communication

```bash
python tools/tp_comm_estimator.py \
  --message-mib 16 --bandwidth-gbs 400 \
  --latency-us 5 --collectives-per-layer 2 --layers 80
```

详细说明：[tools/README.md](tools/README.md)

---

## 学完以后至少要能回答

### Transformer / Inference

- 为什么 Attention 要除以 `sqrt(d_head)`？
- GQA 为什么能显著降低 KV Cache？
- 为什么 KV Cache 并没有让 Decode Attention 变成 O(1)？
- 为什么小 Batch Decode 经常是 Memory-bound？
- 为什么 Batch 增大会提高 Throughput，却可能伤害 TPOT/P99？

### GPU

- Arithmetic Intensity 是什么？
- 为什么 100% Occupancy 仍然可能很慢？
- GEMV 与 GEMM 的硬件行为有什么区别？
- Kernel Fusion 到底省了什么？
- 为什么 FlashAttention 的核心是 IO-aware？

### Distributed

- 手推 2-way Tensor Parallel MLP。
- 为什么 TP 更适合高速 intra-node interconnect？
- PP Bubble 是怎么产生的？
- 为什么 MoE Expert Parallelism 天然对应 AllToAll？
- 什么情况下 Communication 真正可以和 Compute overlap？

### Serving

- Continuous Batching 和 Static Batching 的区别？
- PagedAttention 解决的究竟是什么问题？
- Chunked Prefill 为什么可以保护 Decode TPOT？
- Prefix Cache Locality 与 Load Balance 为什么可能冲突？
- P/D Disaggregation 后会产生什么新瓶颈？

### Profiling

- TTFT 很高但 TPOT 正常，你先看什么？
- GPU Utilization 低，如何系统缩小问题范围？
- Nsight Systems 与 Nsight Compute 分别回答什么问题？
- Throughput 提升但 P99 变差，这算优化成功吗？

---

## 这个仓库不重点覆盖

为了保持主线清晰，目前不试图系统覆盖：

- C++ / Python 基础语法；
- CUDA 入门语法；
- 通用后端八股；
- Kubernetes / MLOps 全家桶；
- RAG / Vector DB 应用开发；
- 大模型 Pretraining / Post-training 算法大全；
- 所有最新推理论文。

目标不是“什么都有”，而是让一个具备编程基础的人用尽量短的路径建立真正连贯的：

> **Model → Inference → GPU → Distributed → Serving → Runtime → Profiling**

系统认知。

---

## Contributing

欢迎补充：

- 更短、更准确的机制解释；
- Worked Example；
- 真实 AI Infra 面试题；
- vLLM / SGLang / Megatron 架构更新；
- Profiling case study；
- 公式修正；
- 小型 calculator / visualization。

但尽量不要简单堆链接。一个新资源最好能够回答：

> **它相比仓库已有资料，究竟在哪个知识点上讲得更好？**

详细规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。
