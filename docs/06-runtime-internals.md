# 06 — Runtime Internals: vLLM / SGLang

## Goal

Do not memorize every class name. Build a stable mental model of a modern inference engine.

## Generic request lifecycle

```text
HTTP / API request
      ↓
input processing / tokenization
      ↓
request state enters engine
      ↓
scheduler selects work for next step
      ↓
KV-cache manager allocates / reuses cache blocks
      ↓
batch metadata is built
      ↓
model runner executes model/kernels
      ↓
logits / sampling
      ↓
request state updated; token streamed
      ↓
repeat until finished
```

In distributed deployments add workers, collective communication, routing and possibly P/D or expert-parallel stages.

## vLLM topics

You should be able to locate and conceptually understand:

- engine/core boundary;
- waiting/running request state;
- scheduler policy and token budgeting;
- KV-cache manager / block allocation;
- prefix-cache lookup;
- model runner;
- attention backend;
- sampling/output path;
- tensor/pipeline/data/expert parallel execution;
- speculative decoding path;
- preemption and memory pressure behavior.

### Best starting point

**Inside vLLM: Anatomy of a High-Throughput LLM Inference System**  
https://vllm.ai/blog/2025-09-05-anatomy-of-vllm

Then keep the actual repository open while reading:  
https://github.com/vllm-project/vllm

## SGLang topics

Use SGLang as a comparison point rather than learning two systems independently.

Focus on:

- RadixAttention / prefix caching;
- scheduler and overlap ideas;
- continuous batching;
- paged/token attention;
- speculative decoding;
- structured output path;
- TP/PP/EP/DP;
- prefill/decode disaggregation;
- cache-aware scheduling/routing.

Repository: https://github.com/sgl-project/sglang

## Source-reading method

When opening a large runtime, trace **one request**, not the whole repository.

1. Find request admission.
2. Find request state object.
3. Find scheduler entrypoint.
4. Find where cache capacity is checked/allocated.
5. Find where the batch handed to the worker is constructed.
6. Find the model-runner call.
7. Find attention/KV metadata.
8. Find sampling/output update.
9. Put breakpoints/logging around those transitions.

Only after you can trace the happy path should you inspect preemption, prefix caching, speculative decoding or distributed paths.

## Interview prompts

- What data structures must an inference scheduler track?
- Why is CPU scheduler overhead visible at very fast decode rates?
- What state is per request vs per sequence vs global cache state?
- How does a runtime represent non-contiguous KV cache physically/logically?
- Where would you implement a new scheduling policy?
- Where would you measure queueing time vs execution time?
- How would you add a new trace field without perturbing critical-path performance too much?
- What changes when moving from one GPU to tensor-parallel multi-GPU serving?
