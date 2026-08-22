# 03 — GPU Performance Mental Model

> **Goal:** answer “why is this workload slow on a GPU?” with a structured model of execution, memory movement, occupancy, and arithmetic intensity.

For AI Infra interviews, CUDA syntax is less important than being able to map an operator to hardware behavior.

---

## 1. The hierarchy

A simplified CUDA execution hierarchy:

```text
grid
└── thread block / CTA
    └── warps
        └── threads
```

A GPU contains many Streaming Multiprocessors (SMs). Thread blocks are scheduled onto SMs; warps are the hardware scheduling unit.

On NVIDIA GPUs, a warp contains 32 threads.

### SIMT

Threads in a warp execute in a SIMT style: same instruction stream, different data.

If threads take different branches:

```cpp
if (threadIdx.x % 2 == 0) {
    path_A();
} else {
    path_B();
}
```

both paths may need to be executed for different lanes, reducing effective parallelism. This is **warp divergence**.

---

## 2. Why GPUs tolerate latency

HBM access is far slower than arithmetic.

The GPU hides latency by keeping many warps ready:

```text
warp 0 waits for memory
→ scheduler runs warp 1
→ warp 1 waits
→ scheduler runs warp 2
...
```

This is why parallelism and occupancy matter.

But:

> **High occupancy is a means to hide latency, not the optimization objective itself.**

A kernel can have 100% occupancy and still be slow because it is bandwidth-bound, instruction-bound, poorly vectorized, or doing unnecessary work.

---

## 3. Memory hierarchy

A useful conceptual hierarchy:

```text
registers             fastest, per-thread
shared memory / L1    on-chip, per-SM
L2 cache              shared across SMs
HBM / global memory   large, high bandwidth, high latency
```

Exact capacities and bandwidths vary by architecture.

The systems question is always:

> **How many times does data cross the slow boundary?**

This is the intuition behind tiling, fusion, FlashAttention, quantization, and cache reuse.

---

## 4. Coalesced global memory access

A warp should ideally access neighboring addresses so hardware can combine requests into efficient memory transactions.

Good pattern:

```text
thread 0 → A[0]
thread 1 → A[1]
thread 2 → A[2]
...
```

Bad/strided pattern:

```text
thread 0 → A[0]
thread 1 → A[1024]
thread 2 → A[2048]
...
```

The second pattern may require many memory transactions.

Interview phrasing:

> Coalescing is about turning lane-level memory requests from a warp into a small number of aligned global-memory transactions.

---

## 5. Shared memory and tiling

Suppose many threads need the same tile of a matrix.

Instead of repeatedly reading it from HBM:

```text
HBM → thread
HBM → thread
HBM → thread
...
```

load once into shared memory:

```text
HBM → shared-memory tile
          ↓
      reused many times
```

This raises arithmetic intensity by performing more computation per HBM byte.

### Tiled GEMM intuition

For:

```text
C = A × B
```

a block loads tiles of A and B into shared memory and performs many multiply-accumulate operations using those tiles before loading the next ones.

That reuse is one reason matrix multiplication can achieve very high GPU utilization.

---

## 6. Shared-memory bank conflicts

Shared memory is divided into banks.

If threads in a warp access addresses that map to different banks, accesses can proceed efficiently.

If many threads hit the same bank in conflicting ways, requests may serialize.

You do not need to memorize every architecture's bank layout for a general Infra interview. You should know the principle:

> **Shared memory is fast only when the access pattern is friendly.**

---

## 7. Registers and register pressure

Registers are very fast but finite per SM.

A kernel using many registers per thread may reduce the number of simultaneously resident warps/blocks.

So an optimization can backfire:

```text
more per-thread registers
→ fewer resident warps
→ less latency hiding
```

This is called register pressure.

Again, trade-offs matter more than maximizing a single metric.

---

## 8. Occupancy

Occupancy is roughly:

```text
active warps / maximum supported active warps
```

Resident blocks/warps are constrained by resources such as:

- registers;
- shared memory;
- max threads;
- max blocks/warps per SM.

Low occupancy can hurt latency hiding.

But high occupancy does not guarantee speed.

### Interview example

If a kernel is already saturating HBM bandwidth, increasing occupancy may not improve performance because the bottleneck is external memory bandwidth.

---

## 9. Roofline model

The single most useful performance model for AI Infra:

```text
Arithmetic Intensity (AI) = FLOPs / bytes transferred
```

Approximate attainable performance:

```text
Performance ≤ min(
    Peak Compute,
    Memory Bandwidth × Arithmetic Intensity
)
```

This produces two regimes:

```text
low AI  → bandwidth-bound
high AI → compute-bound
```

### Ridge point

The boundary is approximately:

```text
ridge AI = peak FLOPs/s / peak bytes/s
```

If an operation's arithmetic intensity is far below the ridge point, buying more compute does little unless memory traffic is reduced.

---

## 10. GEMV vs GEMM

### GEMV-like workload

```text
y = W x
```

The matrix W may be huge but each weight participates in very little computation per invocation.

Low reuse → low arithmetic intensity.

### GEMM-like workload

```text
Y = X W
```

with many rows in X.

The same weights are reused across many rows, so arithmetic intensity is higher.

This is the key intuition behind small-batch vs large-batch decode.

```text
small batch decode → more GEMV-like
large batch decode → more GEMM-like
```

---

## 11. Operator classification

A useful first-pass table:

| Operator | Typical shape intuition | Likely concern |
|---|---|---|
| Large GEMM | high reuse | compute / Tensor Cores |
| Small GEMM/GEMV | low reuse | memory bandwidth |
| RMSNorm | reduction + elementwise | bandwidth + reduction |
| Softmax | reductions + exp | bandwidth + synchronization |
| Embedding lookup | irregular reads | memory latency/bandwidth |
| Decode attention | read growing KV | KV bandwidth + kernels |
| MoE dispatch | communication/data movement | network/NVLink + imbalance |

Never treat the table as universal. Use profiler evidence.

---

## 12. Kernel launch overhead

Launching a GPU kernel is not free.

A model may contain many very small operations:

```text
kernel A  3 μs
kernel B  5 μs
kernel C  4 μs
CPU launch gaps
...
```

Even if each kernel is efficient, launch and synchronization overhead can become a large fraction of decode latency.

This motivates:

- kernel fusion;
- CUDA Graphs;
- persistent kernels in some designs;
- runtime/scheduler optimization.

At very high token rates, CPU overhead can become a real serving bottleneck.

---

## 13. Kernel fusion

Suppose execution is:

```text
read x → RMSNorm → write y
read y → activation → write z
```

A fused kernel may perform:

```text
read x
→ RMSNorm
→ activation
→ write z
```

Benefits can include:

- fewer HBM round trips;
- fewer intermediate tensors;
- fewer kernel launches.

But fusion can also increase register pressure or reduce flexibility, so bigger fused kernels are not automatically better.

---

## 14. Tensor Cores

Tensor Cores accelerate supported matrix/tensor operations at specific data types and layouts.

For Infra reasoning, know:

- dense GEMMs can use specialized matrix units;
- tile/layout/alignment matter;
- FP16/BF16/TF32/FP8/INT variants have different support by generation;
- low utilization may come from shapes or kernels that fail to map well to Tensor Cores.

You do not need to memorize every SKU's throughput table unless the role is hardware/performance-specialized.

---

## 15. Streams and overlap

CUDA streams allow operations to be ordered independently, subject to dependencies.

Potential overlap:

```text
GPU compute on stream A
||
communication / memcpy on stream B
```

Overlap can hide some communication or data movement.

But only if dependencies permit it.

If compute needs the communication result immediately:

```text
comm → compute
```

there is no useful overlap for that dependency.

This becomes central in TP/EP systems.

---

## 16. CUDA Graphs: conceptual view

Dynamic Python/C++ launch logic incurs per-step overhead.

CUDA Graphs capture a repeated sequence of GPU operations and replay it with lower launch overhead.

They are attractive for repetitive decode workloads, but dynamic shapes, variable batches, memory addresses, and control flow can complicate capture/reuse.

The interview takeaway:

> CUDA Graphs target CPU/launch overhead, not HBM bandwidth or arithmetic complexity.

---

## 17. FlashAttention: why it is fast

Naive attention conceptually does:

```text
Q,K
 ↓
S = QKᵀ
 ↓
write S to HBM
 ↓
read S
 ↓
P = softmax(S)
 ↓
write/read P
 ↓
P V
```

For long sequences, the `[S,S]` intermediate is large.

FlashAttention changes the schedule:

```text
load Q/K/V tiles
→ compute partial scores on chip
→ maintain online softmax statistics
→ accumulate output
→ avoid materializing full S×S attention matrix in HBM
```

The central idea is **IO awareness**.

### Online softmax intuition

If scores arrive in tiles, keep running values such as:

```text
m = running maximum
l = running normalization sum
o = running weighted output
```

When a new tile changes the max, rescale the old partial sums appropriately.

This lets exact softmax be computed incrementally.

### Interview answer

A strong 30-second answer:

> FlashAttention is faster mainly because it reorganizes exact attention to reduce HBM traffic. It tiles Q/K/V, computes score and softmax blocks using fast on-chip memory, and uses online softmax so the full S×S attention matrix does not need to be materialized in HBM.

---

## 18. Worked Roofline example

Imagine a kernel performs:

```text
2 TFLOPs
```

and moves:

```text
100 GB
```

Arithmetic intensity:

```text
AI = 2e12 / 100e9 = 20 FLOP/byte
```

Suppose hardware has:

```text
peak compute = 100 TFLOP/s
memory BW = 2 TB/s
```

Bandwidth roof:

```text
2 TB/s × 20 FLOP/byte = 40 TFLOP/s
```

So the Roofline upper bound is:

```text
min(100, 40) = 40 TFLOP/s
```

This workload is bandwidth-limited under this simplified model.

To improve it, you would try to reduce bytes moved or increase reuse before focusing on peak compute.

---

## 19. How to reason about an unfamiliar kernel

Use this sequence:

```text
1. What are the tensor shapes?
2. How many FLOPs?
3. How many bytes must be read/written?
4. Is there reusable data?
5. Does access coalesce?
6. Are there large reductions/synchronizations?
7. Is the kernel too small to fill the GPU?
8. Does it map to Tensor Cores?
9. What would nsys/ncu show if my hypothesis is correct?
```

That last question is important: convert intuition into something falsifiable.

---

## 20. Common traps

### “Low GPU utilization means the GPU is weak.”

It may mean CPU launch gaps, underfilled batches, synchronization, communication, or tiny kernels.

### “High occupancy means optimized.”

No. A memory-saturated kernel can already be optimal at lower occupancy.

### “Shared memory is always faster.”

Only if the reuse and access pattern justify moving data there.

### “FlashAttention is faster because it computes less attention.”

It mainly reduces expensive memory IO while preserving exact attention.

### “More fusion is always better.”

Fusion can increase registers, code complexity, compilation cost, or reduce scheduling flexibility.

---

## 21. Interview questions

1. Thread vs warp vs block vs SM?
2. What is warp divergence?
3. How does a GPU hide memory latency?
4. What is occupancy and when does it matter?
5. What is coalesced memory access?
6. Why use shared-memory tiling?
7. What is a bank conflict?
8. What is register pressure?
9. Define arithmetic intensity.
10. Derive the Roofline ridge point.
11. Why is GEMV often bandwidth-bound?
12. Why does larger batch make decode more GEMM-like?
13. Why can kernel fusion help?
14. What problem do CUDA Graphs target?
15. Why is FlashAttention IO-aware?
16. What profiler evidence would show a bandwidth-bound kernel?
17. What profiler evidence would show CPU launch overhead?

---

## Short resources

- GPU MODE lectures: https://github.com/gpu-mode/lectures
- GPU MODE resource stream: https://github.com/gpu-mode/resource-stream
- BBuf CUDA optimization notes: https://github.com/BBuf/how-to-optim-algorithm-in-cuda
- GPU performance engineering resources: https://github.com/wafer-ai/gpu-perf-engineering-resources
- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Nsight Compute: https://developer.nvidia.com/nsight-compute

---

## Definition of done

Given RMSNorm, softmax, GEMM, decode attention, or an unfamiliar kernel, you can:

- estimate its computation/data-movement shape;
- predict a likely compute vs bandwidth regime;
- explain how tiling/fusion/reuse could help;
- identify occupancy/register/launch concerns;
- say what `nsys` or `ncu` evidence would validate the hypothesis.

**Next:** [04 — Distributed Parallelism](04-distributed-parallelism.md)
