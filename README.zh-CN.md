# AI Infra 101

> **从 Transformer 内部机制到 LLM Inference / Serving 面试的最短学习路径。**

这个仓库面向已经具备基本 Python / C++ / CUDA 能力，但想系统建立 **AI Infra / LLM Systems** 知识体系的人。

它不要求你完整看一门几十小时的课程，而是把每层知识压缩成：

> **必须掌握什么 → 最值得看的现成资料 → 面试要能解释/推导/手写到什么程度。**

## 知识栈

1. [Transformer 执行原理](docs/01-transformer-execution.md)
2. [LLM Inference](docs/02-llm-inference.md)
3. [GPU Performance](docs/03-gpu-performance.md)
4. [Distributed Parallelism](docs/04-distributed-parallelism.md)
5. [LLM Serving](docs/05-llm-serving.md)
6. [vLLM / SGLang Runtime](docs/06-runtime-internals.md)
7. [Profiling & Optimization](docs/07-profiling.md)
8. [面试准备](docs/08-interview-prep.md)

配套：

- [常用公式速查](cheatsheets/formulas.md)
- [Collective / Parallelism 速查](cheatsheets/collectives.md)
- [100 道 Infra 自测题](interview/100-questions.md)
- [手撕题 Checklist](interview/handwrite-checklist.md)
- [完整资料索引](resources/reading-list.md)

## 推荐使用方式

不要按“读完”衡量学习进度。每个知识点都问自己四件事：

- **Explain**：能否脱离资料口述 2–3 分钟？
- **Derive**：能否现场推 shape / FLOPs / memory / communication？
- **Implement**：能否写出核心伪代码或 PyTorch 实现？
- **Diagnose**：遇到性能问题，能否提出合理 profiling 路径？

AI Infra 实习面试里，整个 stack 至少要做到 **Explain + Derive**；Transformer / tensor 基础最好达到 **Implement**；Inference / GPU 部分要开始具备 **Diagnose** 能力。

## 这个仓库不重点覆盖

- C++ / Python 基础语法
- CUDA 入门语法
- 通用后端八股
- Kubernetes / MLOps 全家桶
- RAG / Vector DB 应用开发
- 大模型训练算法大全

目标不是“什么都收录”，而是保持 **LLM Systems / Inference / Serving / Runtime / GPU performance** 的主线。

> 英文 README 是项目对外主入口；中文版本更适合快速复习。
