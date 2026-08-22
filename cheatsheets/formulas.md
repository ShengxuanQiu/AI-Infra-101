# AI Infra Formula Cheat Sheet

These are first-order estimation formulas. Real kernels/runtimes add alignment, allocator metadata, temporary buffers, fragmentation, quantization metadata and implementation-specific effects.

## Model weights

```text
weight_bytes ≈ parameter_count × bytes_per_parameter
```

Examples: FP32=4 B, FP16/BF16=2 B, INT8≈1 B, 4-bit≈0.5 B before metadata/packing overhead.

## KV cache

For standard per-layer K/V storage:

```text
KV_bytes = 2 × B × S × L × Hkv × Dh × bytes_per_element
```

`B` concurrency/batch, `S` cached sequence length, `L` layers, `Hkv` KV heads, `Dh` head dim.

Per token per request:

```text
KV_bytes_per_token = 2 × L × Hkv × Dh × bytes_per_element
```

This is why MQA/GQA reduce KV memory: they reduce `Hkv`.

## Attention score complexity

Conceptual dense prefill attention per layer:

```text
QK^T + PV = O(B × S^2 × Hq × Dh)
```

Projection/MLP work scales roughly linearly with `S`; attention interactions scale quadratically with dense sequence length.

## Arithmetic intensity / Roofline

```text
AI = FLOPs / bytes moved
attainable_perf ≈ min(peak_compute, memory_bandwidth × AI)
```

Low AI → likely bandwidth-bound; high AI → more likely compute-bound, assuming sufficient parallelism.

## Memory-bound lower bound

Very rough:

```text
time >= bytes_that_must_move / achieved_bandwidth
```

For batch-1 dense decode, reading model weights can dominate this lower bound. With larger batch, weight reads are amortized across more tokens, while per-request KV traffic grows.

## Throughput / latency

```text
TTFT = request arrival → first output token
TPOT ≈ average time between generated output tokens
E2E = request arrival → completion
```

For output length `O`:

```text
E2E ≈ TTFT + (O - 1) × TPOT
```

This ignores streaming/network/application overhead details but is useful for reasoning.

## Tensor-parallel communication intuition

For ring algorithms with `N` ranks and tensor size `M` bytes, per-rank communicated volume is commonly on the order of:

```text
AllGather      ~ (N-1)/N × M
ReduceScatter  ~ (N-1)/N × M
AllReduce      ~ 2(N-1)/N × M
```

Exact cost depends on algorithm, topology, latency terms and whether you count send, receive, or total link traffic. Use these as order-of-magnitude reasoning tools, not universal timing formulas.

## Performance speedup sanity

```text
speedup = baseline_time / optimized_time
scaling_efficiency ≈ speedup / number_of_devices
```

Always report workload and latency/throughput objective with the number; isolated “x faster” is rarely enough.
