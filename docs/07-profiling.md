# 07 — Profiling & Optimization

> **Goal:** replace “GPU utilization is low” with a falsifiable performance diagnosis.

Performance engineering is not guessing optimizations. It is narrowing a hypothesis space with evidence.

The core loop:

```text
workload → metric → baseline → timeline → bottleneck class
→ deeper profiler → hypothesis → one change → re-measure
```

---

## 1. Define the problem precisely

Bad problem statement:

> vLLM is slow.

Good problem statement:

> On one H100, model X at BF16 with input length 2K, output length 256 and concurrency 32 achieves 1,800 output tok/s, but P99 TPOT is 55 ms while the target is 30 ms.

You want:

- model + precision;
- hardware + topology;
- input/output length distribution;
- concurrency/QPS;
- scheduler settings;
- parallelism configuration;
- exact metric;
- baseline.

Without a stable workload, profiler traces are hard to compare.

---

## 2. Build a baseline before touching code

Record at least:

```text
TTFT
TPOT / ITL
E2E latency
throughput
goodput if SLO exists
GPU memory usage
KV-cache usage
batch size / scheduled tokens
```

Also record software/hardware versions when reproducibility matters.

Do not optimize against a moving workload.

---

## 3. Start coarse, then zoom in

A good profiling funnel:

```text
application metrics
      ↓
runtime/scheduler metrics
      ↓
Nsight Systems timeline
      ↓
hot kernel / collective
      ↓
Nsight Compute / NCCL detail
      ↓
source-level optimization
```

Starting with `ncu` on random kernels is usually inefficient because you may optimize a kernel that is not responsible for the end-to-end problem.

---

## 4. `nvidia-smi`: useful but coarse

It can tell you:

- process memory;
- coarse utilization;
- power;
- clocks/thermals;
- device health.

It cannot tell you why utilization is low.

A 30% utilization reading could mean:

- CPU gaps;
- communication waits;
- bursty kernels;
- insufficient batch size;
- low occupancy;
- workload idle periods.

Use it as a smoke detector, not a diagnosis.

---

## 5. PyTorch Profiler

Useful when you need framework/operator attribution:

- which PyTorch ops consume time;
- CPU vs CUDA activities;
- operator call stacks;
- memory events;
- shapes;
- user annotations.

It bridges Python-level model code and GPU events.

Questions it can answer:

```text
Which operator family is hot?
Is time in attention, GEMM, normalization, sampling, or CPU code?
Are unexpected tensor copies occurring?
```

For full system scheduling/communication timelines, Nsight Systems is usually the next step.

---

## 6. Nsight Systems: the timeline tool

Think of `nsys` as a movie of the system.

Look for:

- CPU threads;
- CUDA API calls;
- kernel launches;
- GPU kernels;
- streams;
- memcpy;
- NCCL;
- synchronization;
- idle gaps;
- NVTX ranges.

### Pattern A — CPU launch bottleneck

Timeline:

```text
GPU kernel
      gap
GPU kernel
      gap
GPU kernel
```

CPU thread may be preparing work slower than the GPU consumes it.

Possible directions:

- reduce Python/control overhead;
- fuse kernels;
- CUDA Graphs;
- async scheduling;
- persistent batch metadata.

### Pattern B — communication-bound

```text
compute
NCCL ─────────────
compute
NCCL ─────────────
```

If GPU compute repeatedly waits for collectives, inspect topology/message sizes and overlap opportunities.

### Pattern C — long-prefill interference

```text
decode kernels
decode kernels
long prefill =================================
decode resumes
```

This can explain TPOT spikes even when average throughput is good.

### Pattern D — host-device synchronization

Repeated synchronization calls can serialize work that could otherwise overlap.

---

## 7. NVTX: make traces readable

Raw traces can contain thousands of kernels.

Annotate high-level phases:

```text
request admission
scheduler
prefill
decode
KV transfer
sampling
```

Then the profiler can answer application questions instead of only showing kernel names.

Good instrumentation is one of the highest-leverage skills in systems work.

---

## 8. Nsight Compute: kernel microscope

Use `ncu` **after** identifying important kernels.

Questions:

- Is memory bandwidth saturated?
- Is compute throughput high?
- Are Tensor Cores used?
- What limits occupancy?
- Are warps stalled on memory/dependencies?
- What is cache behavior?
- Is instruction mix surprising?

Common metric families include:

```text
memory throughput
SM / compute utilization
achieved occupancy
warp stall reasons
L1/L2 hit behavior
Tensor Core utilization
instructions
```

Metric names change by architecture/tool version; understand the concepts rather than memorizing a dashboard.

---

## 9. Diagnosing memory-bound kernels

Hypothesis:

> decode projection is HBM-bandwidth-bound.

Evidence you would expect:

- high achieved DRAM/HBM throughput relative to achievable bandwidth;
- compute units not near peak;
- low arithmetic intensity;
- larger batch increases compute utilization and tokens/s.

Possible optimizations:

- increase reuse/batching;
- lower weight precision;
- fuse operations where applicable;
- reduce unnecessary memory traffic;
- better layout/coalescing.

If memory bandwidth is *not* high, your diagnosis may be wrong: perhaps access is latency-bound, launch-bound, or underfilled.

---

## 10. Diagnosing compute-bound kernels

Evidence:

- high compute/Tensor Core utilization;
- arithmetic intensity above Roofline ridge;
- memory bandwidth below saturation because compute is limiting;
- runtime scales with operation count.

Possible directions:

- use better Tensor Core mapping;
- improve tile shape;
- reduce unnecessary FLOPs;
- use lower-precision compute;
- increase kernel efficiency.

---

## 11. Diagnosing underfilled kernels

A kernel may be neither compute- nor bandwidth-saturated because there is not enough parallel work.

Examples:

- tiny batch;
- very small tensor;
- serial reduction;
- excessive kernel fragmentation.

Evidence:

- short kernels;
- low number of active blocks/warps;
- lots of launch overhead;
- increasing workload size greatly improves utilization.

Potential fixes:

- batching;
- fusion;
- persistent kernels;
- combine requests/operators;
- remove CPU gaps.

---

## 12. Scheduler/runtime bottlenecks

Not every serving bottleneck is inside a kernel.

Possible runtime issues:

### Queueing overload

```text
arrival rate ≈/exceeds service capacity
→ queue time grows
```

### Too-small batches

GPU is repeatedly underfilled.

### Too-large scheduling steps

Throughput improves but latency tails worsen.

### KV pressure

Frequent allocation failures/preemption/recompute.

### CPU scheduling

Engine cannot prepare GPU work fast enough.

### Tokenization/output handling

Can matter when requests are small and decode is fast.

Always correlate profiler traces with runtime metrics.

---

## 13. Distributed bottlenecks

For TP/EP/PP, add:

- NCCL collective duration;
- message sizes;
- link utilization;
- cross-node vs intra-node placement;
- compute/communication overlap;
- stragglers;
- MoE token imbalance.

### TP symptom

Increasing TP from 4→8 reduces per-GPU compute but latency worsens.

Possible reason:

```text
saved compute < added collective cost
```

### EP symptom

Average tokens/expert looks reasonable, but P99 layer latency spikes.

Inspect per-step expert token distribution; a hot expert can create a straggler.

---

## 14. Roofline workflow in practice

For a hot kernel:

1. estimate FLOPs;
2. estimate bytes moved from slow memory;
3. calculate arithmetic intensity;
4. compare to hardware ridge point;
5. check profiler evidence;
6. decide whether to optimize compute or data movement.

This keeps optimization grounded in physics.

---

## 15. Worked example: decode throughput is low

Assume:

```text
GPU utilization: 45%
HBM usage: not full capacity
throughput: lower than expected
```

Do **not** conclude “memory bottleneck” from GPU utilization.

### Step 1 — workload

Check:

```text
batch/concurrency
input/output lengths
scheduler token budget
```

Suppose concurrency = 2.

### Step 2 — `nsys`

You see many short GEMV-like kernels with CPU gaps.

### Hypothesis

GPU is underfilled and launch/control overhead is significant.

### Experiment

Increase concurrency to 32 while holding other conditions constant.

Result:

```text
throughput 4× higher
GPU timeline denser
larger GEMMs
```

Now the bottleneck may move toward memory/compute saturation.

The correct optimization was not a custom CUDA kernel; it was a workload/scheduler issue.

---

## 16. Worked example: TPOT spikes during long prompts

Observation:

```text
median TPOT normal
P99 TPOT spikes correlate with long prompts
```

### Hypothesis

Long prefill monopolizes execution steps.

### Evidence

- runtime trace shows long prefill batches;
- `nsys` shows large prefill kernel region before decodes resume;
- spikes correlate with prompt length.

### Candidate changes

- chunked prefill;
- prefill scheduling limits;
- P/D separation;
- priority for decode work.

Measure both:

```text
P99 TPOT
TTFT of long prompts
throughput
```

because protecting decode may worsen prefill completion.

---

## 17. Worked example: P/D disaggregation is slower

You expected lower interference but see worse E2E latency.

Break down:

```text
prefill queue
prefill compute
KV transfer
wait at decode pool
decode
```

Possible outcomes:

- KV transfer dominates;
- decode pool is undersized;
- routing is imbalanced;
- network shares bandwidth with other traffic;
- prefill/decode ratio does not match workload.

Disaggregation must be profiled as a pipeline, not evaluated by GPU utilization alone.

---

## 18. Optimization experiment discipline

Change one variable where possible.

Bad experiment:

```text
new scheduler
+ new CUDA kernel
+ quantization
+ larger batch
```

If performance changes, you do not know why.

Better:

```text
baseline
→ change chunk size only
→ measure
→ change token budget only
→ measure
```

For systems research, this becomes the basis of ablation studies.

---

## 19. Regression dimensions

An optimization that improves average throughput may regress:

- TTFT;
- TPOT;
- P99;
- memory;
- quality;
- small models;
- long context;
- low concurrency;
- multi-GPU scaling.

Always ask what workload the optimization sacrifices.

---

## 20. A reusable interview diagnosis template

Question:

> Our LLM server is slow. What do you do?

Answer structure:

```text
1. Define “slow”
   TTFT / TPOT / E2E / throughput / P99?

2. Fix workload
   model, precision, lengths, QPS, concurrency, topology

3. Split queue vs service time

4. Inspect runtime metrics
   batch size, scheduled tokens, KV, cache hits, preemption

5. Inspect system timeline (`nsys`)
   CPU gaps / kernels / memcpy / NCCL / synchronization

6. Classify
   underfilled / bandwidth / compute / communication / queueing

7. Zoom into hot kernel if needed (`ncu`)

8. Form one hypothesis

9. Change one mechanism and remeasure all important metrics
```

That structure is itself a strong interview signal.

---

## 21. Common traps

### “nvidia-smi says GPU utilization is 50%, so the kernel is bad.”

Coarse utilization cannot localize the cause.

### “Optimize the hottest kernel first.”

Only if it meaningfully affects the end-to-end metric.

### “High HBM utilization always means optimal.”

You may still be moving unnecessary bytes.

### “Average latency improved, so optimization succeeded.”

P99 or SLO goodput may regress.

### “Profile production with every detailed metric enabled.”

Profilers add overhead. Use controlled reproductions and appropriate sampling/instrumentation.

---

## 22. Interview questions

1. `nvidia-smi` vs PyTorch Profiler vs `nsys` vs `ncu`?
2. Why start with a timeline before kernel metrics?
3. How do you recognize CPU launch overhead?
4. What evidence suggests memory-bandwidth saturation?
5. What evidence suggests compute saturation?
6. What is an underfilled kernel?
7. How would you profile long-prefill interference?
8. How would you profile TP scaling degradation?
9. How would you detect MoE load imbalance?
10. Why can increasing concurrency improve GPU utilization?
11. What does NVTX add to profiling?
12. How do you evaluate a scheduler optimization without confounding variables?
13. What regressions should you check after throughput improves?
14. How would you separate queueing latency from model execution latency?

---

## Short resources

- GPU MODE lectures: https://github.com/gpu-mode/lectures
- GPU MODE resource stream: https://github.com/gpu-mode/resource-stream
- BBuf optimization notes: https://github.com/BBuf/how-to-optim-algorithm-in-cuda
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Nsight Compute: https://developer.nvidia.com/nsight-compute
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

---

## Definition of done

Given a slow LLM workload, you can produce a profiling plan that:

- fixes the workload and metric;
- separates queueing/runtime/GPU/communication causes;
- uses `nsys` before blindly diving into kernels;
- uses `ncu` for targeted kernel hypotheses;
- ties every optimization to expected profiler evidence;
- checks latency, throughput and memory regressions.

**Next:** [08 — Interview Prep](08-interview-prep.md)
