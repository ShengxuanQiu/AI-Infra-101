# 06 — Runtime Internals: vLLM / SGLang

> **Goal:** trace one request through a modern inference engine without memorizing every class name.

Runtime code changes quickly. The stable mental model is:

```text
request admission
→ scheduling
→ KV allocation/reuse
→ batch construction
→ distributed model execution
→ attention/KV kernels
→ logits/sampling
→ request state update
→ streaming output
```

If you can trace this path, new versions of vLLM/SGLang become much easier to understand.

---

## 1. Generic process view

A production runtime usually contains at least three conceptual planes:

```text
Serving / API plane
- HTTP/OpenAI-compatible server
- tokenization / input validation
- streaming responses

Engine control plane
- request state
- scheduler
- KV-cache manager
- batch metadata

Execution plane
- GPU workers
- model runner
- attention backend
- distributed collectives
- kernels
```

Modern implementations may place these in different processes/threads, but the responsibilities remain similar.

---

## 2. Request lifecycle

A useful end-to-end trace:

```text
1. request arrives
2. prompt is tokenized / validated
3. engine creates request state
4. request enters waiting queue
5. scheduler chooses tokens to execute
6. KV manager checks/reuses/allocates blocks
7. runtime constructs scheduled batch metadata
8. worker/model runner executes forward pass
9. attention backend reads/writes KV
10. logits are sampled
11. request records new token / cache progress
12. token is streamed to client
13. repeat until stop condition
14. release KV and request state
```

Every optimization you learn lives somewhere on this path.

---

## 3. Per-request state vs global state

### Per-request state

Typical fields conceptually include:

```text
request id
prompt token ids
output token ids
num computed tokens
sampling params
priority / arrival time
finished / stop state
allocated KV references
prefix-cache state
```

### Global scheduler state

```text
waiting requests
running requests
scheduler policy
token budget
max sequences
available KV capacity
possibly encoder/multimodal budgets
```

### Global cache state

```text
physical KV block pool
free block list
logical→physical mappings
prefix hash/index
reference counts
possibly offload/tiering state
```

Keeping these layers separate makes source reading much easier.

---

## 4. The scheduler is the engine's decision maker

At each iteration, the scheduler decides **what work runs next**.

Conceptual pseudocode:

```python
def schedule():
    budget = max_tokens_per_step
    scheduled = []

    # Continue running requests / decodes.
    for req in running:
        if can_schedule(req, budget, kv_capacity):
            n = tokens_to_run(req)
            reserve_kv(req, n)
            scheduled.append((req, n))
            budget -= n

    # Admit waiting requests / prefills.
    for req in waiting:
        cached = lookup_prefix(req)
        needed = remaining_prefill(req, cached)
        chunk = min(needed, budget)
        if can_allocate(req, chunk):
            scheduled.append((req, chunk))
            budget -= chunk

    return scheduled
```

Real schedulers are more complex, but this captures the important resources:

```text
token budget + sequence slots + KV capacity + policy
```

---

## 5. Current vLLM mental model

vLLM's modern V1 architecture separates an API/entrypoint side from an **Engine Core** and GPU worker/model-runner execution. The official architecture docs are the best source for current process details:

- Architecture overview: https://docs.vllm.ai/en/latest/design/arch_overview/
- Scheduler configuration: https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- Scheduler implementation/API: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/

The stable path to learn is:

```text
API / AsyncLLM
      ↓
Engine Core
      ↓
Scheduler ─────→ KVCacheManager
      ↓
Executor / Worker
      ↓
ModelRunner
      ↓
Model + attention backend + kernels
```

Exact classes can change. Focus on responsibilities.

---

## 6. vLLM scheduler: what to inspect

The current scheduler configuration exposes concepts such as:

- maximum batched/scheduled tokens;
- maximum sequences;
- scheduling policy;
- chunked prefill;
- partial/long prefill controls;
- async scheduling;
- KV-related admission/reserve behavior.

This tells you something important about production runtimes:

> **Scheduling is constrained by tokens and memory, not just request count.**

When reading source, locate:

```text
waiting queue
running list
per-step token budget
prefill admission
KV allocation failure
preemption
scheduled request metadata
```

---

## 7. KVCacheManager: logical vs physical state

A paged KV manager conceptually handles:

```text
request logical tokens
       ↓
logical cache blocks
       ↓ block table
physical GPU cache blocks
```

Responsibilities include:

- allocate blocks as sequences grow;
- free blocks when sequences finish;
- track block ownership/reference counts;
- identify reusable prefix blocks;
- expose physical mappings needed by attention kernels;
- possibly coordinate offload/transfer.

The scheduler asks the cache manager questions like:

```text
How many cached tokens already exist?
Can I reserve N additional tokens?
Which physical blocks belong to this request?
Can blocks be reclaimed/preempted?
```

---

## 8. Prefix-cache lookup path

Conceptually:

```text
prompt token blocks
      ↓ hash/key
prefix index
      ↓
matching physical/cache blocks
      ↓
num cached tokens
```

If a request has a cache hit:

```text
prompt length = 4096
cached prefix = 3072
new prefill work = 1024 tokens
```

The runtime still needs request metadata that links the reused blocks into the request's logical cache sequence.

A source-reading exercise is to find:

1. where prefix hashes are created;
2. where lookup occurs;
3. where reference counts/ownership change;
4. how the scheduler changes `num_computed_tokens` or equivalent state.

---

## 9. ModelRunner: where control becomes tensor work

The scheduler operates on request/token metadata.

The model runner transforms that into GPU execution.

Conceptually it must prepare:

- token IDs / positions;
- sequence lengths;
- slot/block mappings;
- attention metadata;
- sampling metadata;
- distributed-worker state;
- persistent batch structures or graph-capture-compatible buffers.

Then:

```text
model.forward(...)
→ attention backend
→ kernels
→ logits
```

This is a useful boundary:

> **Scheduler speaks in requests and token budgets; ModelRunner speaks in tensors, indices, layouts, and device buffers.**

---

## 10. Attention metadata

Paged attention needs more than Q/K/V tensors.

The kernel needs to know where each sequence's KV lives.

Typical conceptual metadata:

```text
sequence lengths
block tables
slot mappings
query lengths
context lengths
positions
```

During decode:

```text
new token's K/V
→ write to physical slot
query
→ traverse block table
→ read historical K/V blocks
```

This is where a logical sequence becomes non-contiguous physical memory accesses.

---

## 11. The persistent batch idea

A naive runtime might rebuild every tensor/list for the active batch each iteration.

At high decode rates, CPU metadata work can become expensive.

Optimized runtimes often maintain persistent or reusable batch structures and update only changed slots.

Systems benefit:

- fewer allocations/copies;
- lower Python/CPU overhead;
- more stable addresses for graph capture;
- lower per-token scheduler overhead.

This is a reminder that once GPU kernels get very fast, **control-plane cost becomes visible**.

---

## 12. Sampling/output path

After model execution:

```text
hidden states
→ LM head / logits
→ sampling transformation
→ next token id
```

Runtime then:

- appends token to request state;
- checks EOS/stop strings/max tokens;
- sends/streams output;
- decides whether request remains running.

At high throughput, even output serialization and detokenization can become measurable CPU work.

---

## 13. Preemption path

Suppose KV allocation fails.

Conceptually:

```text
scheduler wants more blocks
→ KV manager reports insufficient capacity
→ choose victim/request policy
→ release or move state
→ request returns to waiting/preempted state
→ later recompute/resume
```

When reading code, trace:

- who selects the victim;
- what request progress is retained;
- what KV references are released;
- how recomputation amount is determined;
- how repeated thrashing is avoided.

Recent vLLM scheduler controls explicitly expose admission/reserve mechanisms intended to avoid excessive cache thrashing; current behavior should always be checked against the latest official docs.

---

## 14. KV transfer / disaggregated serving

In prefill/decode disaggregation:

```text
prefill engine produces KV
      ↓
KV connector / transfer layer
      ↓
decode engine consumes KV
```

A runtime now needs state machines for:

- remote KV availability;
- transfer completion;
- source/destination block mapping;
- failures/timeouts;
- scheduling only when dependencies are satisfied.

This is substantially more complex than local cache allocation.

---

## 15. Distributed runtime path

For tensor-parallel serving:

```text
scheduler builds one logical batch
        ↓
workers on TP ranks execute corresponding shards
        ↓
collectives at layer boundaries
        ↓
logits/sampling owner or synchronized result
```

Questions to answer from source:

- which process owns scheduler state?
- how is metadata broadcast to workers?
- which rank samples?
- where do TP collectives occur?
- how are workers synchronized?
- what happens if one worker fails?

For EP/DP/P-D combinations, process topology becomes even more important.

---

## 16. SGLang as a comparison

Do not learn vLLM and SGLang as two unrelated codebases.

Map the same concepts:

```text
request
→ scheduler
→ memory/KV pool
→ batch
→ model worker
→ attention/token kernels
→ output
```

SGLang is especially useful for studying:

- RadixAttention / prefix reuse;
- scheduling overlap;
- continuous batching;
- paged/token attention;
- structured-output integration;
- speculative decoding;
- distributed serving.

Repository: https://github.com/sgl-project/sglang

Docs: https://docs.sglang.ai/

The exact implementation evolves, so use official docs/source for current details.

---

## 17. Source-reading strategy that actually works

Do **not** open a 100k-line runtime and read directories alphabetically.

Trace one request.

### Pass 1 — happy path

```text
API request
→ request object
→ waiting queue
→ scheduler
→ KV allocation
→ model runner
→ sampling
→ output
```

### Pass 2 — state ownership

For each state ask:

```text
who creates it?
who mutates it?
CPU or GPU?
per request or global?
lifetime?
```

### Pass 3 — exceptional paths

Only then inspect:

- prefix-cache hit;
- preemption;
- speculative decoding;
- P/D transfer;
- distributed modes;
- multimodal inputs.

---

## 18. How to instrument a runtime

Suppose you want per-request queue time.

Bad approach:

```text
add printf() everywhere on critical path
```

Better:

1. record a monotonic timestamp at request admission;
2. record first scheduled timestamp;
3. store compact numeric fields in request state;
4. emit asynchronously or at request completion;
5. sample/disable instrumentation for production if overhead matters.

For GPU timing, use existing tracing/NVTX/profiler infrastructure rather than CPU wall time around asynchronous launches.

---

## 19. Where would you implement a scheduling policy?

Before coding, identify:

```text
policy input:
- waiting/running requests
- token budget
- KV capacity
- priority/SLO/cache locality

policy output:
- selected requests
- number of tokens/chunks to run
- preemptions/admissions
```

Then keep execution mechanisms unchanged if possible.

Good systems design separates:

```text
policy
from
mechanism
```

This makes experiments safer and easier to evaluate.

---

## 20. Worked trace

Request:

```text
prompt = 3000 tokens
max_new_tokens = 200
```

Assume 2000 prompt tokens hit prefix cache and scheduler has a 1024-token prefill chunk budget.

### Step A

```text
cache hit: 2000
remaining prefill: 1000
schedule: 1000 prefill tokens
allocate KV for new tokens
run model
request now fully prefetched
sample first output token
```

### Step B

```text
schedule: 1 decode token
allocate one KV slot per relevant layer
run model with Q length 1 and context ≈ 3001
sample output token
```

### Later

Repeat decode until completion, then free/request-release non-shared KV references.

A good source-reading exercise is to locate every state transition in the actual runtime.

---

## 21. Common traps

### “The scheduler directly runs CUDA kernels.”

Usually the scheduler chooses work; model-runner/worker layers translate it into device execution.

### “PagedAttention is just a cache-manager feature.”

Both the logical allocation layer and attention kernel must understand the paged representation.

### “Only GPU code matters in inference.”

At fast decode rates, CPU scheduling, tokenization, launch, IPC and output processing can matter.

### “Learning class names equals understanding architecture.”

Class names change. State ownership and execution flow are durable.

---

## 22. Interview questions

1. What state must an inference scheduler track?
2. What belongs per-request vs globally?
3. Why does a KV manager need both logical and physical mappings?
4. What metadata does paged attention need?
5. Where does prefix-cache lookup affect scheduling?
6. What does the model runner do that the scheduler does not?
7. Why can CPU overhead become visible in decode?
8. What happens on KV allocation failure?
9. Where would you implement a new scheduling policy?
10. How would you measure queueing time accurately?
11. How would you add a low-overhead trace field?
12. What new runtime state appears with P/D disaggregation?
13. What changes in multi-GPU TP execution?
14. How would you trace one request through an unfamiliar runtime?

---

## Short resources

- vLLM architecture overview: https://docs.vllm.ai/en/latest/design/arch_overview/
- vLLM scheduler docs: https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- vLLM source: https://github.com/vllm-project/vllm
- Inside vLLM: https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- SGLang: https://github.com/sgl-project/sglang
- SGLang docs: https://docs.sglang.ai/

---

## Definition of done

Open the current vLLM or SGLang source and trace one request from API admission to output. You should be able to identify:

- request state;
- waiting/running queues;
- scheduler decision;
- KV allocation/reuse;
- batch/model-runner boundary;
- attention metadata;
- sampling/update path;
- release/preemption path.

**Next:** [07 — Profiling & Optimization](07-profiling.md)
