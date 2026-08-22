# Tiny AI Infra Calculators

These scripts are intentionally simple. They are designed for **back-of-the-envelope reasoning and interview practice**, not production capacity planning.

## KV-cache calculator

```bash
python tools/kv_cache_calculator.py \
  --layers 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --seq-len 16384 \
  --bytes-per-element 2 \
  --concurrency 16 \
  --kv-budget-gib 32
```

It uses:

```text
KV bytes = 2 × layers × sequence × KV_heads × head_dim × bytes/element
```

Use it to sanity-check GQA, long-context, quantized-KV, and concurrency questions.

---

## Roofline estimator

```bash
python tools/roofline_estimator.py \
  --flops 2e12 \
  --bytes 1e11 \
  --peak-tflops 100 \
  --bandwidth-gbs 2000
```

Outputs:

- arithmetic intensity;
- hardware ridge point;
- bandwidth roof;
- first-order compute/bandwidth regime;
- idealized lower-bound time.

Use it to practice reasoning about GEMM/GEMV, attention, normalization, and kernel optimization.

---

## TP communication estimator

```bash
python tools/tp_comm_estimator.py \
  --message-mib 16 \
  --bandwidth-gbs 400 \
  --latency-us 5 \
  --collectives-per-layer 2 \
  --layers 80 \
  --overlap 0.25
```

Uses the intentionally simplified model:

```text
T_collective ≈ latency + bytes / effective_bandwidth
```

This is **not** a replacement for NCCL benchmarking. It is a teaching tool for understanding why frequent per-layer communication can dominate TP scaling.

---

## Why tiny scripts?

The goal is not to build a simulator. The goal is to make common AI Infra calculations executable so you can:

1. derive the formula yourself;
2. check your arithmetic;
3. vary one assumption;
4. build intuition about sensitivity.

If a calculator gives a surprising result, return to the symbolic formula before trusting the number.
