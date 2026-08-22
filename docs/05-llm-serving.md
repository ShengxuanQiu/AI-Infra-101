# 05 — LLM Serving

## Goal

Move from “a model can generate” to “a system can serve many heterogeneous requests under latency and throughput objectives.”

## Metrics

Know the distinction between:

- **TTFT** — time to first output token;
- **TPOT / ITL** — time per output token / inter-token latency;
- **E2E latency** — request completion latency;
- **throughput** — tokens/s or requests/s;
- **goodput** — useful throughput meeting an SLO;
- **P50/P95/P99** — distribution matters, not only averages.

## Core serving mechanisms

### Continuous batching

Do not lock a fixed batch until every sequence completes. At iteration boundaries, finished sequences can leave and waiting sequences can enter, improving utilization under variable request lengths.

### PagedAttention / paged KV management

The key systems problem is dynamic KV-cache allocation and fragmentation. Paging-like blocks let cache storage be managed non-contiguously, enabling denser memory use and higher concurrency.

**Important interview distinction:** this primarily attacks memory-management waste/concurrency; it does not magically make a single request's attention arithmetic disappear.

### Chunked prefill

Break long prefill work into chunks so the scheduler can mix prompt processing with decode work rather than allowing a huge prefill to monopolize a step. Think about head-of-line blocking, TTFT/TPOT trade-offs and scheduling budgets.

### Prefix caching

Reuse KV state for shared prefixes. This introduces cache admission/eviction and cache-aware routing/scheduling questions.

### Prefill/Decode disaggregation

Prefill and decode have different resource characteristics. Separating them can improve resource specialization and interference control, but now KV state must move between stages and the system needs routing/load-balancing decisions.

### Speculative decoding

Draft several candidate tokens cheaply, verify them with the target model, accept a prefix, and reduce serial target-model steps when acceptance is high enough.

### Quantization

Understand the serving impact of weight precision and KV precision: memory footprint, bandwidth, kernel support, accuracy and dequantization/compute trade-offs.

## Scheduler questions

A serving scheduler reasons about constraints such as:

- token budget;
- KV capacity;
- waiting/running queues;
- prefill vs decode work;
- priority/fairness;
- preemption/recomputation/swap;
- prefix locality;
- SLO/tail latency.

## Recommended resources

- **Inside vLLM** — best single walkthrough for a modern high-throughput runtime:  
  https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- **vLLM PagedAttention intro**:  
  https://vllm.ai/blog/2023-06-20-vllm
- **GPU MODE awesomeMLSys** — Orca, PagedAttention, Sarathi/chunked prefill, DistServe, speculative decoding:  
  https://github.com/gpu-mode/awesomeMLSys
- **SGLang** — RadixAttention, prefix caching, scheduler, distributed serving:  
  https://github.com/sgl-project/sglang
- **Modular LLM Inference Handbook**:  
  https://github.com/modular/llm-inference-handbook
- **AI-Infra** — broad serving/caching/orchestration notes:  
  https://github.com/pacoxu/AI-Infra

## Paper landmarks

You do not need to memorize every paper. Know the problem → mechanism → trade-off of these landmarks:

- **Orca** — iteration-level scheduling / continuous batching lineage;
- **PagedAttention / vLLM** — KV memory management;
- **FlashAttention** — IO-aware exact attention kernels;
- **Sarathi-Serve** — chunked prefill / stall-free scheduling lineage;
- **DistServe** — prefill/decode disaggregation and SLO-aware goodput;
- **SGLang** — prefix-aware runtime / RadixAttention;
- **Speculative decoding** — reducing serial target-model decoding steps.

## Interview prompts

- Why is static batching inefficient for variable-length generation?
- Why does PagedAttention increase concurrency?
- What are the trade-offs of larger scheduling token budgets?
- A long prefill hurts ongoing chat TPOT. What mechanisms help?
- When does prefix caching have a high hit rate?
- Why can cache-aware routing conflict with load balancing?
- What new bottleneck appears after P/D disaggregation? (Hint: KV transfer + stage balance.)
- How would you optimize for throughput vs tail latency differently?
