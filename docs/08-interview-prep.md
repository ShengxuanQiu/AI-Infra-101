# 08 — AI Infra Interview Prep

> **Goal:** turn scattered knowledge into answers you can derive, implement, and defend under interview pressure.

AI Infra interviews vary by team, but most questions fall into four modes:

```text
Explain     → Can you explain the mechanism precisely?
Derive      → Can you calculate shapes/memory/FLOPs/communication?
Implement   → Can you write the core operation correctly?
Diagnose    → Can you debug a systems problem from evidence?
```

You should train all four.

---

## 1. What is usually tested?

For internship / junior LLM systems roles:

1. Transformer / LLM internals;
2. tensor and PyTorch implementation;
3. inference math and memory;
4. GPU architecture/performance intuition;
5. distributed parallelism;
6. serving/runtime mechanisms;
7. profiling/debugging;
8. project deep-dive;
9. sometimes general algorithms and C++ concurrency.

Weighting depends on the team:

| Team | Extra emphasis |
|---|---|
| Inference / serving | KV cache, scheduling, batching, vLLM/SGLang, TTFT/TPOT |
| Kernel / performance | CUDA, memory hierarchy, reductions, GEMM, FlashAttention, profiling |
| Distributed training/inference | NCCL, TP/PP/DP/EP, overlap, topology |
| ML systems research | paper reasoning, experiments, bottleneck identification, prototypes |
| Accelerator / architecture | Roofline, memory hierarchy, dataflow, workload characterization |

---

## 2. The 60-second answer format

For a concept question, use:

```text
1. Definition
2. Why it exists / problem it solves
3. Mechanism
4. Main trade-off
5. One systems consequence
```

Example: **What is GQA?**

```text
Definition:
Grouped-query attention uses more query heads than KV heads.

Problem:
Full MHA stores separate K/V for every query head, making KV cache large.

Mechanism:
Groups of query heads share one KV head.

Trade-off:
Less KV flexibility than full MHA, but much lower serving state.

Systems consequence:
KV bytes scale with Hkv, so GQA reduces cache capacity and bandwidth pressure.
```

This is much stronger than a one-line textbook definition.

---

## 3. The derivation format

For estimation questions:

```text
1. Define symbols
2. Write symbolic formula
3. Substitute values
4. Check units
5. State ignored factors
```

Example: KV cache.

```text
KV bytes
= 2 × layers × sequence × KV heads × head dim × bytes/element
```

Then plug numbers.

Finish with:

> This is the raw tensor storage; allocator block granularity, workspace, metadata and other runtime buffers add overhead.

Interviewers usually prefer a transparent approximation over fake precision.

---

## 4. Three kinds of hand-write

### A. Whiteboard pseudocode — mandatory

Reconstruct without API lookup:

- stable softmax;
- RMSNorm;
- scaled dot-product attention;
- causal mask;
- MHA/GQA mapping;
- RoPE;
- SwiGLU;
- KV-cache update;
- top-k/top-p sampling;
- simple MoE router.

### B. PyTorch implementation — strongly recommended

Write a small correct module using basic tensor ops in ~10–20 minutes.

The interviewer is checking:

- shape discipline;
- broadcasting;
- numerical stability;
- correct masking;
- state/cache handling;
- ability to debug.

### C. CUDA/Triton — role dependent

General inference roles:

- understand reduction/softmax/GEMM optimization patterns.

Kernel roles:

- actual CUDA/Triton implementation;
- memory coalescing;
- shared-memory tiling;
- occupancy/register trade-offs;
- benchmarking/profiling.

Practice list: [handwrite-checklist.md](../interview/handwrite-checklist.md).

---

## 5. What “hand-write attention” should mean

At minimum you should be able to produce:

```python
q = x @ Wq
k = x @ Wk
v = x @ Wv

q = reshape_heads(q)
k = reshape_heads(k)
v = reshape_heads(v)

scores = q @ k.transpose(-1, -2) / sqrt(head_dim)
scores += causal_mask
probs = stable_softmax(scores)
out = probs @ v
out = merge_heads(out)
out = out @ Wo
```

Then answer:

- shapes at each line;
- complexity;
- what changes for GQA;
- what changes for decode KV cache;
- why FlashAttention does not materialize the full score matrix.

Writing syntax without being able to explain these is not enough.

---

## 6. Core derivations to practice

You should derive these from memory.

### Transformer

- attention tensor shapes;
- rough parameters/layer;
- GEMM FLOPs `≈ 2MKN`;
- MHA/GQA KV dimensions.

### Inference

- weight memory;
- KV bytes/token/request;
- rough concurrency from HBM budget;
- bandwidth lower bound `bytes/BW`;
- effect of batch on weight bytes per generated token.

### GPU

- arithmetic intensity;
- Roofline upper bound;
- ridge point.

### Distributed

- two-way column/row TP;
- TP MLP communication;
- rough communication time `α + bytes/BW`;
- PP stage bottleneck;
- MoE dispatch volume intuition.

These derivations turn “memorized knowledge” into reusable reasoning.

---

## 7. Project deep-dive: the most important section

Your resume project is not a presentation. Expect adversarial follow-ups.

Prepare:

```text
Problem
→ why it mattered

Baseline
→ exact system / workload

Architecture
→ where your code sits

Bottleneck
→ evidence, not intuition

Change
→ mechanism / code path

Evaluation
→ workload + metrics + baselines

Trade-off
→ where it loses / fails

Ownership
→ what exactly you implemented

Next step
→ what you would change now
```

### Bad project answer

> We optimized vLLM and got 20% faster.

### Strong project answer

> Decode TPOT regressed under long-prefill bursts. Runtime traces showed decode iterations blocked behind large prefill steps rather than a kernel bottleneck. I changed the scheduling policy to cap/chunk prefill work, then evaluated TTFT, TPOT and throughput across prompt-length distributions. P99 TPOT improved, but long-prompt TTFT increased, so the final policy exposed the trade-off through a configurable token budget.

The second answer exposes systems thinking.

---

## 8. Source-level project questions

If you say you modified vLLM/SGLang/CUDA, expect:

- Which file/class/function?
- What thread/process owned the state?
- What data structure changed?
- Was it on the critical path?
- How did you avoid synchronization/race issues?
- CPU or GPU memory?
- What happened on preemption/failure?
- How did you measure overhead?
- What regression tests did you run?
- Why not implement it somewhere else?

Do not put source-level claims on your resume that you cannot defend.

---

## 9. System-design question: design an LLM serving system

Use this order.

### Step 1 — workload

Ask:

```text
model size / architecture?
precision?
prompt length distribution?
output length distribution?
QPS / concurrency?
bursty or steady?
shared prefixes?
streaming?
```

### Step 2 — SLO

```text
TTFT target?
TPOT target?
P99?
throughput/goodput target?
```

### Step 3 — back-of-envelope sizing

Estimate:

```text
weight memory
KV/request
rough concurrency
parallelism requirement
```

### Step 4 — execution layout

Choose:

```text
single GPU / TP / PP / DP / EP
```

based on model and topology.

### Step 5 — runtime mechanisms

Only now discuss:

- continuous batching;
- paged KV;
- chunked prefill;
- prefix caching;
- quantization;
- speculation;
- P/D disaggregation.

Tie every mechanism to a workload constraint.

### Step 6 — overload/failure

Discuss:

- queue limit;
- admission control;
- replica failure;
- KV transfer failure;
- timeout/retry;
- observability.

### Step 7 — metrics

Explain how you would validate the design.

---

## 10. Debugging questions: think like a narrowing tree

Question:

> GPU utilization is low. Why?

Do not list 20 causes randomly.

Use categories:

```text
Is there enough work?
├─ low QPS / low batch
├─ tiny shapes
└─ load imbalance

Is GPU being fed?
├─ CPU scheduler/tokenization
├─ kernel launch gaps
└─ synchronization

Is GPU waiting on data?
├─ NCCL/network
├─ memcpy/KV transfer
└─ storage/offload

Are kernels inefficient?
├─ memory access
├─ occupancy/registers
├─ poor Tensor Core mapping
└─ many tiny kernels
```

Then say what profiler evidence distinguishes branches.

---

## 11. Common high-frequency questions and answer skeletons

### Why is decode often memory-bound?

```text
one/few new tokens
→ low weight reuse
→ large weights read every layer
→ growing KV reads
→ low arithmetic intensity
```

Add the caveat: larger batch/hardware/quantization can move the regime.

### Why does GQA help serving?

```text
KV bytes ∝ Hkv
→ fewer KV heads
→ less persistent state and KV traffic
→ higher context/concurrency budget
```

### Why does FlashAttention help?

```text
naive attention materializes large intermediates in HBM
→ tiled IO-aware schedule + online softmax
→ less HBM traffic
→ exact attention
```

### Why does TP scale poorly sometimes?

```text
less compute/GPU
but
more/frequent communication + synchronization
→ when communication > saved compute, latency worsens
```

### Why does PagedAttention help?

```text
variable-length KV causes allocation/fragmentation waste
→ fixed blocks + block table + paged kernel
→ better effective memory utilization/concurrency
```

---

## 12. Behavioral questions for systems candidates

Even research/Infra interviews often probe engineering behavior.

Prepare examples for:

- hardest performance bug;
- disagreement over a systems design;
- experiment whose hypothesis was wrong;
- debugging nondeterministic behavior;
- optimizing something that caused a regression;
- reading/modifying an unfamiliar large codebase;
- deciding what **not** to optimize.

Strong answers use concrete evidence and trade-offs.

---

## 13. General coding / C++ preparation

Do not ignore standard coding screens.

For general AI Infra roles, be comfortable with:

- arrays/hash maps/heaps;
- BFS/DFS;
- intervals/two pointers;
- binary search;
- LRU cache;
- producer-consumer;
- thread-safe queue;
- mutex/condition variable;
- atomic basics;
- RAII/smart pointers;
- move semantics at conceptual level.

But if you already have strong C++ fundamentals, prioritize Infra-specific derivations and tensor code rather than grinding hundreds of unrelated problems.

---

## 14. Mock interview format

A useful 60-minute mock:

```text
10 min — resume/project deep dive
10 min — Transformer/inference fundamentals
10 min — GPU/performance
10 min — distributed/serving
10 min — hand-write problem
10 min — system/debugging question
```

Score each answer on:

```text
correctness
precision
ability to derive
trade-off awareness
communication clarity
```

If you cannot explain an answer in two minutes, the mental model is probably not yet organized.

---

## 15. Self-grading rubric

### Level 0 — recognition

> I have heard of PagedAttention.

Not interview-ready.

### Level 1 — definition

> It uses paging for KV cache.

Still weak.

### Level 2 — mechanism

> Variable-length KV is allocated in fixed blocks with logical-to-physical mappings; kernels use block tables.

Good foundation.

### Level 3 — derive/trade-off

> I can quantify block waste, explain fragmentation, and state that attention still reads historical KV.

Interview-ready.

### Level 4 — diagnose/design

> Given workload/SLO/cache pressure, I can decide whether paging, prefix caching, chunking, or admission control is actually relevant and say how I would measure it.

Strong systems signal.

---

## 16. Practice repositories

- TorchCode: https://github.com/duoan/TorchCode
- TorchLeet: https://github.com/Exorust/TorchLeet
- Awesome LLM System Design: https://github.com/neurarch-ai/awesome-llm-system-design
- GPU MODE lectures: https://github.com/gpu-mode/lectures

---

## 17. Practice inside this repo

- [100+ question bank](../interview/100-questions.md)
- [Core 1-minute answers](../interview/core-answers.md)
- [Hand-write checklist](../interview/handwrite-checklist.md)
- [Formula sheet](../cheatsheets/formulas.md)
- [Collectives sheet](../cheatsheets/collectives.md)

---

## Definition of done

You can handle a mock interview where the interviewer changes the numbers or assumptions and you still derive the answer rather than reciting it.

That means:

- Transformer operations are reconstructable;
- inference formulas are derivable;
- GPU bottlenecks are explainable with Roofline/memory hierarchy;
- TP/PP/EP communication can be reasoned from tensor placement;
- serving mechanisms are tied to concrete problems;
- profiling answers start from evidence;
- every major resume project survives source-level follow-ups.
