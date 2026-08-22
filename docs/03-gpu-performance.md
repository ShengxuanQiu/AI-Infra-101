# 03 — GPU Performance Mental Model

## Goal

Be able to answer **“why is this workload slow on a GPU?”** with a structured hardware model.

## Must know

### Execution model

- thread / warp / block (CTA) / grid;
- Streaming Multiprocessor (SM);
- SIMT execution and divergence;
- warp scheduling and latency hiding;
- occupancy as a resource/latency-hiding concept, not a universal objective.

### Memory hierarchy

- registers;
- shared memory;
- L1 / L2;
- HBM/global memory;
- coalesced global loads/stores;
- shared-memory bank conflicts;
- cache locality and data reuse.

### Performance model

```text
Arithmetic Intensity = FLOPs / bytes moved
Attainable performance ≈ min(peak compute, bandwidth × arithmetic intensity)
```

Use this to classify a kernel/workload as primarily compute- or bandwidth-limited.

### Kernel-level concepts

- launch overhead and many-tiny-kernel problems;
- fusion;
- tiling;
- shared-memory reuse;
- vectorized/coalesced access;
- register pressure;
- occupancy;
- Tensor Cores;
- CUDA streams and overlap;
- CUDA Graphs at a conceptual level.

## FlashAttention: the interview explanation

The key insight is **IO awareness**. A naive implementation materializes/intermediately moves a large attention-score matrix through HBM. FlashAttention changes the algorithmic schedule using tiling and online softmax so more intermediate work stays in fast on-chip memory, reducing HBM traffic while computing exact attention.

You should explain:

1. what data would otherwise be written/read;
2. why HBM traffic matters;
3. how tiling changes the IO pattern;
4. that the result is exact attention (up to normal floating-point behavior), not an approximation.

## Recommended resources

- **GPU MODE lectures** — practical material on profiling, compute/memory architecture, CUDA optimization, FlashAttention, NCCL, Tensor Cores:  
  https://github.com/gpu-mode/lectures
- **GPU MODE resource stream** — excellent curated link collection:  
  https://github.com/gpu-mode/resource-stream
- **BBuf CUDA / AI Infra optimization notes** — CUDA kernels, Triton, CUTLASS/CuTe, PyTorch internals and LLM optimization:  
  https://github.com/BBuf/how-to-optim-algorithm-in-cuda
- **GPU performance engineering resources** — useful tiered map of profiling and optimization material:  
  https://github.com/wafer-ai/gpu-perf-engineering-resources
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
- NVIDIA Nsight Compute: https://developer.nvidia.com/nsight-compute

## Interview prompts

- What is a warp and what happens under branch divergence?
- Why can 100% occupancy still be slow?
- What does coalescing mean physically?
- Why does shared memory help tiled GEMM?
- What is a bank conflict?
- What increases arithmetic intensity?
- GEMV vs GEMM: why do they stress hardware differently?
- Why might small-batch decode look GEMV-like?
- Why can kernel fusion improve both bandwidth use and launch overhead?
- Why is FlashAttention an IO optimization?

## Definition of done

Given an operator (RMSNorm, softmax, GEMM, decode attention), you can make a first-pass prediction about compute vs memory behavior, identify likely limiting resources, and state what profiler evidence would confirm or reject that hypothesis.
