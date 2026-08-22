# 02 — LLM Inference

## Goal

Understand why autoregressive inference is a **different systems workload** from training.

## The two phases

### Prefill

The prompt is processed in parallel. Large GEMMs can reuse weights across many prompt tokens, so arithmetic intensity can be high. Prefill creates the initial KV cache and dominates TTFT for long prompts when queueing is small.

### Decode

The model emits one new token per active sequence per step. Each step has limited token-level work while still reading model weights and growing KV state. At small/medium batch sizes this commonly has low arithmetic intensity and is often bandwidth-limited.

Do not turn “prefill compute-bound / decode memory-bound” into a universal law. Model shape, batch, context, quantization, hardware and kernels can shift the bottleneck.

## KV cache

Per request, approximately:

```text
KV bytes = 2 × L × S × Hkv × Dh × bytes_per_element
```

For batch/concurrency `B`, multiply by `B`.

Where:

- `2` = K + V;
- `L` = number of Transformer layers;
- `S` = cached tokens;
- `Hkv` = number of KV heads;
- `Dh` = head dimension.

You should be able to derive this formula, not just quote it.

## Must know

- autoregressive generation loop;
- prefill vs decode;
- KV caching and cache growth;
- MHA/MQA/GQA impact on cache;
- batching and weight reuse;
- static vs continuous batching;
- prefix caching;
- sliding-window attention/cache;
- KV quantization/offload at a conceptual level;
- weight quantization and why memory/computation effects differ;
- speculative decoding: draft → verify → acceptance;
- why speculative decoding can increase useful tokens per expensive target-model step;
- sampling basics: temperature, top-k, top-p.

## Estimation skills

Given model parameters and hardware, estimate:

1. weight memory;
2. KV memory per token / request;
3. maximum rough concurrency under a memory budget;
4. bandwidth lower bound for a memory-bound decode step;
5. why larger batches amortize weight reads but increase KV traffic;
6. why long context eventually changes the memory balance.

## High-signal resources

- **JAX Scaling Book — Inference** — probably the best compact source for inference FLOP/memory/bandwidth reasoning:  
  https://github.com/jax-ml/scaling-book/blob/main/inference.md
- **Modular LLM Inference Handbook** — practical memory and serving explanations:  
  https://github.com/modular/llm-inference-handbook
- **vLLM: PagedAttention intro** — motivation from KV-cache fragmentation and serving throughput:  
  https://vllm.ai/blog/2023-06-20-vllm
- **Inside vLLM** — modern end-to-end view of inference runtime:  
  https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- **AI-Infra caching notes (Chinese-friendly companion)**:  
  https://github.com/pacoxu/AI-Infra/blob/main/docs/inference/caching.md

## Interview prompts

- Calculate KV cache for a 32-layer model with 8 KV heads, 128 head dim, BF16 and 16K context.
- Why does GQA change KV memory but keep many query heads?
- A 70B model fits in HBM but concurrency is poor. What memory consumers do you inspect?
- Why can increasing batch size improve decode throughput?
- Why can the same change hurt TPOT or tail latency?
- What does prefix caching reuse exactly?
- How does speculative decoding trade extra draft compute for fewer target-model serial steps?
