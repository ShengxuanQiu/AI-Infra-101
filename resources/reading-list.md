# Annotated Reading List

This list is intentionally selective. The goal is not “all AI Infra links”; it is **high-signal material that replaces long course-watching**.

## 0. Broad maps

### GPU MODE — awesomeMLSys
https://github.com/gpu-mode/awesomeMLSys

**Use for:** paper/repo onboarding across attention, serving, speculative decoding, quantization, sparsity and distributed systems.  
**Why:** one of the highest-signal compact reading lists in ML systems.

### pacoxu/AI-Infra
https://github.com/pacoxu/AI-Infra

**Use for:** broad AI-infrastructure landscape, especially as a Chinese-language companion.  
**Caution:** much broader than the interview-focused scope of this repo; use it to branch out, not as a linear syllabus.

### AI Infra Performance Learning
https://github.com/ai-infra-curriculum/ai-infra-performance-learning

**Use for:** checking whether you missed a category (GPU, profiling, distributed inference, deployment).  
**Caution:** its full curriculum is much larger than the compact path here.

---

## 1. Transformer execution

### ruisp666/transformer-from-scratch
https://github.com/ruisp666/transformer-from-scratch

**Covers:** RoPE, RMSNorm, SwiGLU, GQA, KV cache, sparse MoE.  
**Use for:** implementation reference and property-based sanity checks.

### bkitano/llama-from-scratch
https://github.com/bkitano/llama-from-scratch

**Covers:** a readable progression from Transformer components toward Llama-style architecture.  
**Use for:** quick refresh when a component feels fuzzy.

### TorchCode
https://github.com/duoan/TorchCode

**Covers:** auto-graded PyTorch implementation tasks including softmax, RMSNorm, attention, GQA, RoPE and inference-related problems.  
**Use for:** interview hand-writing practice rather than passive reading.

### TorchLeet
https://github.com/Exorust/TorchLeet

**Use for:** additional PyTorch/LLM interview implementation drills.

---

## 2. Inference math & memory

### JAX Scaling Book — inference
https://github.com/jax-ml/scaling-book/blob/main/inference.md

**Covers:** inference memory, FLOPs, bandwidth, batching and KV-cache reasoning with formulas.  
**Use for:** the quantitative mental model behind “prefill vs decode”.

### Modular LLM Inference Handbook
https://github.com/modular/llm-inference-handbook

**Covers:** practical inference memory and serving calculations.  
**Use for:** quick lookup and worked intuition.

### vLLM PagedAttention introduction
https://vllm.ai/blog/2023-06-20-vllm

**Covers:** why KV-cache memory management becomes a serving bottleneck and how paging-inspired allocation helps.

---

## 3. GPU performance

### GPU MODE lectures
https://github.com/gpu-mode/lectures

**Covers:** profiling, GPU compute/memory architecture, CUDA performance, reductions, FlashAttention, NCCL, Tensor Cores and modern kernel topics.  
**Use for:** open the slide/notebook for exactly the concept you need; do not watch everything sequentially.

### GPU MODE resource stream
https://github.com/gpu-mode/resource-stream

**Use for:** curated books/blogs/papers when you want one deeper reference for a GPU concept.

### BBuf/how-to-optim-algorithm-in-cuda
https://github.com/BBuf/how-to-optim-algorithm-in-cuda

**Covers:** CUDA kernels, Triton, CUTLASS/CuTe, PTX, PyTorch internals, LLM inference/training optimization.  
**Use for:** practical Chinese/English notes and code-oriented deep dives.

### wafer-ai/gpu-perf-engineering-resources
https://github.com/wafer-ai/gpu-perf-engineering-resources

**Use for:** tiered list of GPU performance material, especially Roofline/profiling/modern-kernel references.

### NVIDIA profiling tools
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Nsight Compute: https://developer.nvidia.com/nsight-compute

---

## 4. Distributed LLM systems

### Stas Bekman — ML Engineering / model parallelism
https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism

**Covers:** intuitive Tensor Parallelism and distributed memory/performance considerations.  
**Use for:** the first explanation before diving into Megatron source.

### NVIDIA Megatron-LM
https://github.com/NVIDIA/Megatron-LM

**Covers:** TP, PP, EP, CP, distributed optimizer and communication overlap.  
**Use for:** canonical implementation details and current distributed-Transformer mechanisms.

### Hugging Face Tensor Parallelism docs
https://github.com/huggingface/transformers/blob/main/docs/source/en/tensor_parallelism.md

**Use for:** quick row/column-parallel diagrams and code-level intuition.

### NVIDIA NCCL
https://github.com/NVIDIA/nccl

**Use for:** collective communication primitives and real GPU communication stack.

---

## 5. LLM serving & runtime

### Inside vLLM: Anatomy of a High-Throughput LLM Inference System
https://vllm.ai/blog/2025-09-05-anatomy-of-vllm

**Covers:** engine/core, scheduler, KV manager, PagedAttention, continuous batching, prefix caching, speculative decoding, P/D disaggregation, multi-GPU serving.  
**Use for:** first runtime deep dive. Read this before blindly browsing vLLM source.

### vLLM
https://github.com/vllm-project/vllm

**Use for:** trace the real request path after building the mental model.

### SGLang
https://github.com/sgl-project/sglang

**Covers:** RadixAttention, prefix-aware runtime, continuous batching, speculative decoding, paged attention, TP/PP/EP/DP, P/D disaggregation and modern serving.  
**Use for:** compare design choices against vLLM; especially prefix-heavy workloads and scheduling.

### GPU MODE awesomeMLSys
https://github.com/gpu-mode/awesomeMLSys

**Use for landmark papers:** Orca, PagedAttention, Sarathi-Serve, DistServe, speculative decoding, FlashAttention.

---

## 6. Interview question banks

### TorchCode
https://github.com/duoan/TorchCode

Best for implementation practice with automatic checking.

### TorchLeet
https://github.com/Exorust/TorchLeet

Good complementary LLM/PyTorch hand-writing set.

### Awesome LLM System Design
https://github.com/neurarch-ai/awesome-llm-system-design

**Covers:** interview-oriented KV cache, inference serving, cost, agents and end-to-end LLM system design.  
**Use for:** practicing structured design answers and depth follow-ups.

### AI Engineer Interview Questions
https://github.com/ombharatiya/AI-Engineer-Interview-Questions

**Use for:** broad question coverage; selectively use inference/production sections rather than memorizing all answers.

---

## How to evaluate a new resource

Before adding a link, ask:

1. Does it teach a mechanism, not just list buzzwords?
2. Is it technically accurate and reasonably current?
3. Does it add something not already covered better elsewhere?
4. Can a learner use it in <1 hour for a focused topic?
5. Is the original source / license clear?

This repository links to external resources; it does not vendor third-party prose by default.
