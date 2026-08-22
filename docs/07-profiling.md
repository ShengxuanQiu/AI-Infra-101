# 07 — Profiling & Optimization

## Goal

Replace “GPU utilization is low” with a falsifiable performance diagnosis.

## The workflow

```text
1. Define metric / workload
2. Reproduce reliably
3. Establish a baseline
4. System-level timeline
5. Classify bottleneck
6. Zoom into hot kernels / communication
7. Form hypothesis
8. Change one thing
9. Re-measure + check regressions
```

## Tools and what they answer

### `nvidia-smi`

Coarse health/utilization/memory/power view. Useful signal, not a diagnosis.

### PyTorch Profiler

Operator-level attribution, CPU/GPU activities, memory and framework-level hotspots.

### Nsight Systems (`nsys`)

Use for the **timeline**:

- CPU gaps;
- kernel launch cadence;
- stream overlap;
- memcpy;
- NCCL;
- synchronization;
- idle GPU periods;
- many tiny kernels;
- request/batch phase behavior when annotated.

### Nsight Compute (`ncu`)

Use for **kernel-level** analysis:

- achieved memory throughput;
- compute utilization;
- occupancy/resource limits;
- warp stalls;
- cache behavior;
- instruction mix;
- Tensor Core use;
- Roofline-style interpretation.

## Common LLM-serving bottleneck classes

- queueing overload;
- CPU scheduling/tokenization overhead;
- GPU underfilled batch;
- memory bandwidth saturation;
- kernel launch overhead;
- poor fusion;
- KV-cache capacity/fragmentation;
- long-prefill interference;
- collective communication;
- network topology;
- host-device synchronization;
- load imbalance (especially MoE / distributed replicas).

## Recommended resources

- GPU MODE lectures: https://github.com/gpu-mode/lectures
- GPU MODE resource stream: https://github.com/gpu-mode/resource-stream
- BBuf optimization notes: https://github.com/BBuf/how-to-optim-algorithm-in-cuda
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
- NVIDIA Nsight Compute: https://developer.nvidia.com/nsight-compute

## Interview diagnosis template

Question: “Our LLM serving throughput is low. What do you do?”

A strong answer should first pin down:

1. model + precision + hardware;
2. input/output length distribution;
3. offered load / concurrency;
4. latency objective;
5. batch/scheduler configuration;
6. TP/PP/EP topology;
7. cache usage;
8. whether the problem is TTFT, TPOT, throughput or P99.

Then:

- use `nsys`/runtime metrics to separate CPU idle, GPU idle, compute, bandwidth and communication;
- only then inspect selected kernels with `ncu`;
- propose one optimization tied to evidence.

The interview signal is not knowing every metric name. It is **a disciplined narrowing process**.
