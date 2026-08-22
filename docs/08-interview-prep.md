# 08 — AI Infra Interview Prep

## What is usually tested?

For internship / junior LLM systems roles, the common surface is:

1. Transformer/LLM internals;
2. tensor/PyTorch implementation;
3. inference math and memory;
4. GPU architecture/performance intuition;
5. distributed parallelism;
6. LLM serving/runtime mechanisms;
7. debugging/profiling;
8. project deep-dive;
9. sometimes general algorithms/LeetCode and C++ concurrency.

The weighting varies by team. Kernel teams push harder on CUDA; serving teams push harder on scheduling/KV/distributed systems; research teams push harder on papers and experimental reasoning.

## Three kinds of “hand-write”

### A. Whiteboard pseudocode — mandatory

Be able to reconstruct the mechanism without API lookup.

Examples:

- stable softmax;
- RMSNorm;
- attention;
- causal mask;
- GQA mapping;
- KV-cache update;
- top-k / top-p sampling;
- simple MoE routing.

### B. PyTorch implementation — strongly recommended

Write a correct small module in ~10–20 minutes using basic tensor ops.

### C. CUDA kernel — role-dependent

For general inference/serving internships, understand reduction/softmax/GEMM optimization patterns. Kernel/performance roles may expect actual CUDA/Triton implementation and optimization.

## Best practice repositories

- **TorchCode** — auto-graded implementation problems including attention/GQA/RoPE/KV cache and advanced inference topics:  
  https://github.com/duoan/TorchCode
- **TorchLeet** — PyTorch and LLM practice problems:  
  https://github.com/Exorust/TorchLeet
- **Awesome LLM System Design** — serving/cost/KV/cache/system-design interview prompts:  
  https://github.com/neurarch-ai/awesome-llm-system-design

## Project deep-dive template

For each project on your resume, prepare this chain:

```text
Problem → why it mattered
Baseline → what was slow/wrong
System architecture → where your code lives
Bottleneck → evidence
Change → exact mechanism
Evaluation → workload + metrics
Trade-off / failure case
What you would do next
```

If you modified vLLM/CUDA/runtime code, be ready for source-level questions: call path, state, synchronization, memory ownership, measurements and regressions.

## System-design answer structure

For “design an LLM serving system”:

1. clarify workload (model, context, output, QPS, concurrency, SLO);
2. estimate memory/compute;
3. choose hardware/model parallelism;
4. design scheduler/batching/cache;
5. define metrics and overload behavior;
6. add routing/cache locality/disaggregation only when justified;
7. discuss failure modes, observability and trade-offs.

Avoid name-dropping vLLM, PagedAttention or speculative decoding before tying them to a workload constraint.

## Practice

- [100-question bank](../interview/100-questions.md)
- [Hand-write checklist](../interview/handwrite-checklist.md)
- [Formula sheet](../cheatsheets/formulas.md)
