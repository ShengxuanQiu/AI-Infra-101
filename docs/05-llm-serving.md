# 05 — LLM Serving

> **Goal:** move from “the model can generate” to “the system can serve many heterogeneous requests while meeting latency and throughput objectives.”

Serving is not just fast matrix multiplication. It is a resource-allocation problem over:

- request arrivals;
- variable prompt/output lengths;
- GPU compute;
- HBM capacity/bandwidth;
- KV-cache blocks;
- scheduler budgets;
- distributed topology;
- SLOs and fairness.

---

## 1. Metrics first

Never discuss “serving performance” without specifying the metric.

### TTFT — Time To First Token

```text
request arrival
→ queueing
→ prompt preprocessing
→ prefill
→ first output token
```

TTFT is especially sensitive to:

- queueing;
- prompt length;
- prefill scheduling;
- prefix-cache hit rate;
- prefill/ decode interference.

### TPOT / ITL — Time Per Output Token / Inter-Token Latency

Measures the cadence of generated tokens after the first token.

Sensitive to:

- decode batch size;
- weight/KV bandwidth;
- scheduler delays;
- communication;
- long-prefill interference.

### End-to-end latency

```text
arrival → request completion
```

Depends on both TTFT and all subsequent decode steps.

### Throughput

Usually expressed as:

```text
output tokens/s
input+output tokens/s
requests/s
```

Always clarify which.

### Goodput

Throughput that satisfies latency/SLO requirements.

A system can have high raw throughput but poor goodput if most users miss latency targets.

### Tail latency

P50 averages can hide bad experiences. P95/P99 matter under bursty load and heterogeneous request lengths.

---

## 2. The queueing picture

A request's observed latency is approximately:

```text
T_request = T_queue + T_service
```

Even a perfectly optimized GPU cannot rescue a severely overloaded queue.

As offered load approaches service capacity, queueing delay can rise sharply.

This leads to a key Infra habit:

> **Before optimizing kernels, ask whether the system is overloaded.**

---

## 3. Why static batching wastes capacity

Suppose four requests start together:

```text
A:  20 output tokens
B:  40 output tokens
C: 300 output tokens
D: 500 output tokens
```

With a rigid batch, A and B finish early but their slots may remain unused until C/D finish.

```text
step 1:  A B C D
...
step 20: A B C D
step 21: - B C D
step 41: - - C D
...
```

GPU utilization collapses as the batch drains.

---

## 4. Continuous batching

Continuous batching changes the scheduling granularity from whole requests to generation iterations.

At each step:

```text
1. finished requests leave
2. waiting requests may enter
3. active batch is rebuilt
4. one or more tokens of work execute
```

Conceptually:

```text
step 20: A B C D
step 21: E B C D
step 41: E F C D
```

This keeps hardware fuller under variable sequence lengths.

### Trade-offs

A scheduler still needs to decide:

- how much prefill vs decode work to admit;
- maximum batch/token budget;
- priority/fairness;
- cache constraints;
- preemption behavior.

Continuous batching solves wasted slots, not every scheduling problem.

---

## 5. Why KV-cache allocation is hard

Requests have dynamic lengths.

A naive runtime might reserve a contiguous maximum-length region per request:

```text
request A: [used used free free free ...]
request B: [used used used free ...]
```

This wastes memory when actual lengths vary.

Alternatively, variable contiguous allocations can fragment memory over time.

Serving wants:

- incremental allocation;
- high utilization;
- fast append;
- easy reclaim;
- support for many concurrent sequences;
- prefix sharing where possible.

---

## 6. PagedAttention / paged KV management

The key insight is to manage KV state in fixed-size blocks/pages instead of requiring one contiguous physical region per sequence.

Logical sequence:

```text
logical block 0
logical block 1
logical block 2
```

may map physically to:

```text
physical block 17
physical block 4
physical block 91
```

A block table provides the mapping.

### Why this helps

- reduces external fragmentation;
- allocates KV incrementally;
- reclaims blocks when requests finish;
- enables higher effective concurrency;
- supports block-level sharing/caching designs.

### What it does not magically solve

PagedAttention does not eliminate:

- reading historical KV during decode;
- the arithmetic of attention;
- scheduling overhead;
- limited HBM bandwidth.

Its core contribution is **memory management + a kernel capable of operating on paged/non-contiguous KV**.

### Interview distinction

Bad answer:

> PagedAttention makes attention O(1).

Good answer:

> PagedAttention attacks KV allocation/fragmentation so more requests fit efficiently; the attention kernel still needs to consume the relevant historical KV.

---

## 7. Scheduling token budgets

Modern runtimes often constrain scheduled work by token budget rather than only request count.

Example:

```text
max tokens this step = 4096
```

Possible schedule:

```text
32 decode tokens
+ 4064 prefill tokens
```

or:

```text
1024 prefill
+ 128 decode
+ more work from other requests
```

The token budget controls the size/duration of each engine step.

### Large budget

Pros:
- better throughput;
- larger GEMMs;
- fewer scheduler iterations.

Cons:
- long steps may delay decode tokens;
- tail latency can worsen;
- memory pressure increases.

### Small budget

Pros:
- responsive/fine-grained scheduling;
- easier latency control.

Cons:
- smaller kernels;
- more scheduler/launch overhead.

---

## 8. Chunked prefill

A long prompt can create a huge prefill operation.

Without chunking:

```text
long prefill ──────────────────────────────
ongoing decodes wait
```

With chunked prefill:

```text
prefill chunk 1
→ decode step
→ prefill chunk 2
→ decode step
...
```

This lets a scheduler mix work types.

### Why it helps

- reduces head-of-line blocking;
- protects TPOT for active decodes;
- can improve GPU utilization by filling token budgets;
- smooths prefill scheduling.

### Trade-off

Chunking can increase the number of scheduling/kernel steps and may delay completion of one long prefill.

The right chunk size is workload-dependent.

---

## 9. Prefix caching

Suppose many requests share:

```text
system prompt + long common context
```

Their prefix KV states may be identical.

A prefix cache stores/reuses those blocks.

### Cache hit benefit

Instead of recomputing:

```text
shared prefix → prefill again
```

reuse:

```text
shared prefix → existing KV blocks
```

This can save prompt compute and reduce TTFT.

### Prefix-cache systems questions

- exact-match vs block-level matching;
- cache key/hash design;
- block reference counting;
- eviction policy;
- memory reserved for reusable vs active state;
- cache-aware routing across replicas.

### Cache locality vs load balance

Suppose worker A has the prefix cached but is busy; worker B is idle but lacks it.

Routing to A saves prefill but adds queueing.

Routing to B recomputes prefix but may start immediately.

This is a genuine serving trade-off.

---

## 10. Preemption

When KV capacity is insufficient, the scheduler may need to pause work.

Possible strategies include:

- recompute evicted state later;
- swap/offload state;
- reject/admission-control new requests;
- prioritize selected requests.

Preemption is not free.

Recomputation spends compute; swapping spends memory/network bandwidth; blocking harms queueing latency.

A good runtime tries to avoid pathological churn.

---

## 11. Prefill / Decode disaggregation

Prefill and decode have different performance characteristics.

A disaggregated design assigns them to different worker pools:

```text
request
  ↓
prefill worker
  ↓ KV transfer
 decode worker
  ↓
stream tokens
```

### Potential advantages

- hardware specialization;
- less prefill/decode interference;
- separate scaling of each phase;
- more targeted SLO control.

### New costs

- KV state transfer;
- routing complexity;
- stage imbalance;
- extra network dependency;
- failure handling across phases.

### Balance condition

If prefill capacity is much higher than decode capacity:

```text
prefill queue small
→ decode queue explodes
```

and vice versa.

So disaggregation turns one scheduling problem into a multi-stage queueing problem.

---

## 12. Speculative decoding in serving

Speculative decoding changes the decode unit of work.

Instead of one target token per serial step:

```text
draft several
→ verify
→ accept some prefix
```

Potential benefits:

- fewer target-model serial iterations;
- improved single-request or low-batch latency.

Serving complications:

- accepted token count varies per request;
- draft/verify work changes scheduling shapes;
- additional model/state memory may be needed;
- benefit depends on batch and acceptance rate.

A runtime must integrate speculation with KV allocation, scheduling, batching, and output handling.

---

## 13. Quantization in serving

Weight quantization can reduce:

- model footprint;
- weight bandwidth;
- sometimes interconnect traffic.

KV quantization can reduce:

- per-request cache footprint;
- KV bandwidth;
- pressure at long context/high concurrency.

The serving question is always end-to-end:

> Does the lower memory traffic outweigh conversion overhead and kernel limitations while preserving acceptable quality?

---

## 14. Scheduler state

A practical scheduler needs some representation of:

### Per request

- request ID;
- priority;
- prompt progress;
- generated tokens;
- stop state;
- sampling parameters;
- queue time.

### KV/cache state

- allocated blocks;
- cached prefix blocks;
- available capacity;
- ownership/reference counts.

### Global scheduling state

- waiting queue;
- running set;
- token budget;
- maximum sequences;
- prefill/decode policy;
- preemption policy.

This is why serving schedulers are systems code, not just batching loops.

---

## 15. Overload and admission control

At high arrival rate, accepting every request immediately can make everyone's latency terrible.

Systems may need:

- queue limits;
- request rejection;
- priority classes;
- rate limiting;
- deadline/SLO-aware scheduling;
- autoscaling.

A strong system-design answer explicitly discusses overload behavior.

---

## 16. Worked diagnosis: high TTFT, normal TPOT

Observation:

```text
TTFT: bad
TPOT: healthy
```

Likely suspects:

- queueing before admission;
- long prefill;
- long-prefill head-of-line blocking;
- poor prefix-cache hit rate;
- prefill-side communication;
- overloaded prefill workers in P/D architecture.

Less likely as the primary issue:

- steady-state decode bandwidth, because TPOT is normal.

Investigation:

```text
1. split queue time vs prefill execution time
2. bucket by input length
3. inspect scheduler traces
4. inspect prefix-cache hit/miss
5. examine prefill GPU timeline
```

This is much stronger than saying “increase batch size.”

---

## 17. Worked diagnosis: throughput high, P99 terrible

Possible cause:

```text
scheduler uses very large batches/token budget
→ excellent average hardware utilization
→ long scheduling steps / queueing
→ poor tails
```

Other suspects:

- bursty arrivals;
- long prompts;
- unfairness/starvation;
- cache/preemption churn;
- distributed stragglers.

The correct objective may be **goodput**, not raw tokens/s.

---

## 18. Paper landmarks: problem → mechanism → trade-off

### Orca

Problem:
- request-level batching wastes slots during variable-length autoregressive generation.

Mechanism:
- iteration-level scheduling.

Remember:
- scheduling granularity matters.

### vLLM / PagedAttention

Problem:
- dynamic KV allocation/fragmentation limits effective concurrency.

Mechanism:
- paged/block-based KV management + paged attention kernel.

Remember:
- memory management, not magic arithmetic reduction.

### FlashAttention

Problem:
- excessive attention HBM traffic.

Mechanism:
- IO-aware tiled exact attention.

Remember:
- kernel-level IO optimization.

### Sarathi-style chunked prefill

Problem:
- long prefills interfere with ongoing decodes / create scheduling stalls.

Mechanism:
- chunk prefill and co-schedule work more smoothly.

### DistServe-style P/D disaggregation

Problem:
- prefill and decode interfere and have different resource characteristics.

Mechanism:
- separate stages/resources and optimize SLO-aware goodput.

### SGLang / RadixAttention

Problem:
- structured workloads often reuse prefixes.

Mechanism:
- prefix-aware cache/runtime organization and scheduling.

---

## 19. Serving design checklist

For any serving system, ask:

```text
Workload
- model size?
- prompt/output length distribution?
- QPS/concurrency?
- burstiness?
- shared prefixes?

SLO
- TTFT?
- TPOT?
- P99?
- throughput/goodput?

Memory
- weights?
- KV/request?
- workspace?

Scheduling
- continuous batching?
- token budget?
- chunked prefill?
- preemption?

Scale
- TP/PP/DP/EP?
- routing?
- P/D disaggregation?

Observability
- queueing?
- cache metrics?
- GPU timeline?
- per-stage latency?
```

---

## 20. Common traps

### “Maximum throughput is the goal.”

Users care about latency/SLOs; goodput can be more meaningful.

### “PagedAttention makes the attention algorithm faster by changing complexity.”

Its central benefit is KV memory management/concurrency.

### “Prefix caching is always good.”

Cache memory, eviction, routing and load balance matter.

### “P/D disaggregation removes interference for free.”

It adds KV transfer and multi-stage balancing.

### “Chunked prefill always lowers TTFT.”

It may protect decodes but can stretch a single prefill depending on scheduling.

---

## 21. Interview questions

1. TTFT vs TPOT vs E2E latency?
2. Throughput vs goodput?
3. Why does static batching waste slots?
4. What does continuous batching change?
5. What memory problem does PagedAttention solve?
6. Why does paged KV require attention-kernel support?
7. What is scheduling token budget?
8. Why does chunked prefill help TPOT?
9. What does prefix caching reuse?
10. Cache locality vs load balance?
11. What is preemption and why is it expensive?
12. What benefits and costs come with P/D disaggregation?
13. Why can speculative decoding complicate batching?
14. How would you handle overload?
15. High TTFT but normal TPOT: where do you look?
16. High throughput but poor P99: what might be wrong?

---

## Short resources

- Inside vLLM: https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- vLLM PagedAttention intro: https://vllm.ai/blog/2023-06-20-vllm
- vLLM docs: https://docs.vllm.ai/
- SGLang: https://github.com/sgl-project/sglang
- GPU MODE awesomeMLSys: https://github.com/gpu-mode/awesomeMLSys
- Modular LLM Inference Handbook: https://github.com/modular/llm-inference-handbook

---

## Definition of done

Given a workload and SLO, you can reason about:

- queueing vs execution;
- static vs continuous batching;
- token budgets;
- KV allocation and PagedAttention;
- chunked prefill;
- prefix caching;
- preemption;
- P/D disaggregation;
- speculative decoding;
- throughput vs tail-latency trade-offs.

**Next:** [06 — Runtime Internals](06-runtime-internals.md)
