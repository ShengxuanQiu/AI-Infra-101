# 02 — LLM Inference

> **Goal:** understand autoregressive inference as a systems workload: two different execution phases, persistent KV state, batching, memory pressure, and latency/throughput trade-offs.

Training and serving execute the same model but expose very different performance regimes. The central reason is simple:

> **Training processes many tokens in parallel; autoregressive decode repeatedly processes a very small number of new tokens while carrying growing state.**

---

## 1. The autoregressive loop

A decoder-only model generates one token at a time:

```text
prompt
  ↓
prefill all prompt tokens
  ↓
produce next-token distribution
  ↓
sample token t₁
  ↓
decode with cached prefix
  ↓
sample token t₂
  ↓
...
```

Pseudocode:

```python
kv_cache = None

# Prefill
logits, kv_cache = model(prompt_tokens, kv_cache=None)
next_token = sample(logits[:, -1])

# Decode
while not finished:
    logits, kv_cache = model(next_token, kv_cache=kv_cache)
    next_token = sample(logits[:, -1])
```

Two phases emerge naturally: **prefill** and **decode**.

---

## 2. Prefill

During prefill, the entire prompt is known and processed in parallel.

For sequence length `S`:

```text
X: [B, S, D]
```

Large linear layers operate on many token rows at once. Weight matrices loaded from HBM can be reused across many token computations, giving relatively high arithmetic intensity.

### What prefill does

- computes all prompt hidden states;
- computes K/V for every prompt token and every layer;
- writes the initial KV cache;
- performs full causal attention over the prompt;
- determines the first generated token.

### Latency impact

For low queueing, long prompts often increase **TTFT (time to first token)** because the model must process all prompt tokens before generation starts.

But do not memorize “prefill = compute-bound” as a law. The actual bottleneck depends on:

- model dimensions;
- prompt length;
- batch size;
- attention implementation;
- precision;
- hardware;
- distributed communication.

---

## 3. Decode

After prefill, only one new token per active sequence is usually processed each decoding step.

Conceptually:

```text
X_new: [B, 1, D]
```

But each new token must still pass through **all model layers**.

The model also needs historical K/V for attention:

```text
Q_new: [B, Hq, 1, Dh]
K_cache: [B, Hkv, S, Dh]
V_cache: [B, Hkv, S, Dh]
```

Attention becomes roughly:

```text
Q_new @ K_cacheᵀ
```

rather than recomputing old keys/values.

### Why small-batch decode is often bandwidth-bound

At batch size 1 or small batch:

- each layer still needs its large weight matrices;
- there are very few token rows to multiply against those weights;
- weights therefore have limited reuse per decoding step;
- K/V for the existing context must also be read.

This gives low arithmetic intensity compared with large GEMMs.

The useful intuition is:

```text
prefill: many tokens reuse model weights

decode: few new tokens repeatedly stream large weights + growing KV state
```

As batch increases, more sequences share each weight load and decode becomes more GEMM-like, so the bottleneck can shift.

---

## 4. KV cache: why it exists

Without KV caching, generating token `t` would recompute K/V for all previous tokens at every step.

That repeats enormous work.

Instead, once a token's K/V are computed for a layer, they are saved:

```text
token 1 → K₁,V₁ ─┐
token 2 → K₂,V₂  │
...               ├→ persistent KV cache
token t → Kt,Vt ─┘
```

For the next token, only `K_new,V_new` are produced and appended.

Minimal pseudocode:

```python
def decode_attention(x, k_cache, v_cache):
    q = x @ Wq
    k_new = x @ Wk
    v_new = x @ Wv

    k_cache.append(k_new)
    v_cache.append(v_new)

    score = q @ k_cache.T / sqrt(head_dim)
    prob = softmax(score)
    return prob @ v_cache
```

This trades recomputation for persistent memory.

---

## 5. Derive KV-cache memory

For one request:

```text
KV bytes = 2 × L × S × Hkv × Dh × bytes_per_element
```

where:

- `2` = K + V;
- `L` = number of decoder layers;
- `S` = cached sequence length;
- `Hkv` = KV heads;
- `Dh` = head dimension.

### Per-token KV cost

Remove `S`:

```text
KV bytes/token = 2 × L × Hkv × Dh × bytes
```

This is a particularly useful quantity for capacity planning.

### Worked example

Suppose:

```text
L = 32
Hkv = 8
Dh = 128
BF16 = 2 bytes
S = 16,384
```

Per token:

```text
2 × 32 × 8 × 128 × 2
= 131,072 bytes
= 128 KiB/token
```

At 16K tokens:

```text
128 KiB × 16,384
≈ 2 GiB/request
```

That number surprises many beginners: **long-context concurrency can be constrained by KV cache even when the model weights already fit.**

### Why GQA matters

Because KV memory scales with `Hkv`, reducing K/V heads can reduce persistent serving state dramatically.

---

## 6. Weight memory vs KV memory

A useful first approximation for model weights:

```text
weight bytes ≈ parameter_count × bytes_per_weight
```

Examples:

```text
7B parameters × 2 bytes ≈ 14 GB
70B parameters × 2 bytes ≈ 140 GB
```

Real deployments add:

- quantization metadata/scales;
- temporary activations;
- CUDA/NCCL workspace;
- allocator fragmentation;
- runtime buffers;
- graph capture buffers;
- KV cache.

So “weights fit in HBM” does **not** mean the serving workload fits at useful concurrency.

---

## 7. Batch size: why throughput and latency fight

Batching lets multiple requests reuse the same model weights.

Imagine a decode step with one token from each active sequence:

```text
batch 1:   [1, D] @ W
batch 32: [32, D] @ W
```

The second operation has much more computation per weight load and usually better accelerator utilization.

So larger batches can improve:

- tokens/s;
- arithmetic intensity;
- Tensor Core utilization.

But they may hurt:

- queueing latency;
- per-request TPOT;
- tail latency;
- scheduler fairness;
- KV capacity.

The serving objective is therefore not “maximize batch size”; it is to find a batch/scheduling regime that meets latency SLOs while keeping hardware busy.

---

## 8. Static vs continuous batching

### Static batching

A fixed batch starts together and may stay together until all sequences finish.

Problem:

```text
request A:  30 output tokens
request B: 500 output tokens
```

If A's slot cannot be reused until B finishes, capacity is wasted.

### Continuous batching

At generation-iteration boundaries:

```text
finished requests leave
waiting requests enter
active batch changes dynamically
```

This is one of the foundational ideas behind modern high-throughput LLM serving.

Continuous batching is covered in more detail in [05 — LLM Serving](05-llm-serving.md).

---

## 9. Prefix caching

Many requests may share a prefix:

```text
system prompt
+ long document
+ conversation prefix
```

If the runtime already has K/V for that exact prefix, it can reuse them instead of recomputing the prefix.

What is reused?

> **The per-layer KV states corresponding to the shared prefix.**

This can reduce prefill work and TTFT.

But cache reuse introduces systems questions:

- how are prefixes keyed?
- how granular is storage?
- how is cache evicted?
- should routing prefer a worker that already has the prefix?
- does cache-local routing create load imbalance?

So prefix caching quickly becomes a scheduling/routing problem, not merely a model trick.

---

## 10. Sliding-window attention and bounded KV

Some models only attend to the latest `W` tokens in certain layers.

Then cache length for those layers can be bounded by the window:

```text
cached length ≤ W
```

This changes long-context KV growth.

Do not assume every modern model uses the same attention pattern. Model configuration matters for capacity estimates.

---

## 11. Quantization: separate the targets

### Weight quantization

Examples conceptually:

```text
FP16/BF16 → INT8 / FP8 / 4-bit variants
```

Potential benefits:

- lower model memory;
- lower weight bandwidth;
- potentially higher throughput if kernels/hardware are efficient.

Trade-offs:

- numerical quality;
- kernel support;
- dequantization overhead;
- calibration/format constraints.

### KV-cache quantization

This targets a different memory consumer: **persistent attention state**.

Potential benefit:

```text
more context/concurrency per GPU
```

Trade-offs include:

- extra quant/dequant work;
- attention kernel support;
- quality sensitivity.

Do not conflate weight quantization with KV quantization.

---

## 12. Speculative decoding

Autoregressive decoding is serial: the target model normally verifies one next token at a time.

Speculative decoding adds a cheaper proposal mechanism:

```text
draft proposes: a b c d
        ↓
target verifies candidates together
        ↓
accept longest valid prefix
        ↓
advance multiple tokens if acceptance is good
```

The high-level goal is to obtain **more accepted output tokens per expensive target-model step**.

Whether it speeds up depends on:

- draft cost;
- verification cost;
- acceptance rate;
- batch size;
- memory overhead;
- target-model bottleneck;
- implementation quality.

It is not “free parallel decoding.”

### Interview reasoning

If acceptance is low, the draft work becomes waste.

If target decode is already highly batched and compute-efficient, verification dynamics differ from batch-1 latency optimization.

Always reason from workload and hardware.

---

## 13. Sampling basics

The model produces logits `z`.

### Temperature

```text
p = softmax(z / T)
```

- lower `T` sharpens distribution;
- higher `T` flattens it.

### Top-k

Keep only the `k` highest-logit candidates.

### Top-p / nucleus

Sort by probability and retain the smallest set whose cumulative probability exceeds `p`.

Sampling often runs on every active sequence every decode step, so even “small” CPU/GPU work can matter at very high token rates.

---

## 14. A simple bandwidth lower-bound model

Suppose a model has `W` bytes of weights and a decode step effectively streams them once from HBM.

With effective memory bandwidth `BW`:

```text
minimum time ≳ W / BW
```

This is intentionally crude, but powerful.

Example:

```text
weights = 14 GB
bandwidth = 1.4 TB/s
```

Idealized lower bound:

```text
14 GB / 1400 GB/s = 0.01 s = 10 ms
```

Actual execution adds KV traffic, imperfect bandwidth utilization, compute, kernel overhead, and communication.

For batch `B`, roughly the same weight read can produce `B` tokens, so weight bytes per generated token are amortized:

```text
weight bytes/token ≈ W / B
```

KV traffic, however, grows with batch and context.

This explains why decode behavior changes as batch grows.

---

## 15. Worked capacity example

Suppose a GPU has 80 GiB HBM.

Approximate runtime budget:

```text
weights             40 GiB
runtime/workspaces    8 GiB
--------------------------------
remaining KV         32 GiB
```

If each request at your target context uses 2 GiB KV:

```text
rough max active requests ≈ 32 / 2 = 16
```

Real systems need reserve space and block granularity, so actual concurrency will be lower.

This is the correct style of interview reasoning: **start with a transparent approximation, then state what you ignored.**

---

## 16. Common traps

### “Decode is always memory-bound.”

Too absolute. Batch size, model size, quantization, hardware, attention length, kernels, and TP can shift the regime.

### “KV cache reduces attention from O(S²) to O(1).”

Wrong. KV caching avoids recomputing previous K/V. A new query still attends over cached history, so attention work/traffic grows with context.

### “Larger batch always improves serving.”

It may improve throughput while harming latency or KV capacity.

### “If weights fit, the model is deployable.”

Serving needs space for KV, workspace and runtime buffers.

---

## 17. Interview questions

1. Prefill vs decode: what differs in tensor shapes and hardware behavior?
2. Why is small-batch decode commonly bandwidth-limited?
3. Derive KV bytes per token.
4. Derive KV bytes for a full request.
5. Why does GQA reduce KV memory?
6. Why can long context hurt throughput even with KV caching?
7. Why does larger batch improve weight reuse?
8. Why can larger batch worsen TPOT/P99?
9. What exactly does prefix caching reuse?
10. Weight quantization vs KV quantization?
11. Explain speculative decoding without naming a specific algorithm.
12. What determines whether speculative decoding helps?
13. What extra memory consumers should you include besides model weights?
14. If TTFT is bad but TPOT is normal, what inference phase do you inspect first?

---

## Short resources

- JAX Scaling Book — inference: https://jax-ml.github.io/scaling-book/inference/
- Modular LLM Inference Handbook: https://github.com/modular/llm-inference-handbook
- vLLM PagedAttention intro: https://vllm.ai/blog/2023-06-20-vllm
- Inside vLLM: https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- AI-Infra caching notes: https://github.com/pacoxu/AI-Infra

Also use [`../cheatsheets/formulas.md`](../cheatsheets/formulas.md).

---

## Definition of done

Given a model config and a GPU memory/bandwidth budget, you can estimate:

- weight memory;
- KV memory/token/request;
- rough concurrency;
- why batch changes decode efficiency;
- whether a problem is likely prefill-, decode-, memory-, or queue-related;
- how GQA, quantization, prefix caching, or speculative decoding change the serving workload.

**Next:** [03 — GPU Performance](03-gpu-performance.md)
