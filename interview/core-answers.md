# Core AI Infra Interview Answers

> 30 high-frequency questions with compact **1-minute answers**. Do not memorize the wording; use these as answer structures and then derive details yourself.

---

## Transformer execution

### 1. Why scale attention logits by `1/sqrt(d_head)`?

A dot product between random query/key vectors sums roughly `d_head` terms, so its variance grows with `d_head`. Large logits push softmax toward saturation and make gradients/numerics poorly conditioned. Scaling by `1/sqrt(d_head)` keeps the score scale roughly stable as head dimension changes.

### 2. MHA vs MQA vs GQA?

MHA has one K/V head per query head. MQA shares one K/V head across all query heads. GQA is in between: groups of query heads share a K/V head. The important serving consequence is that KV-cache storage scales with the number of KV heads, so MQA/GQA reduce persistent cache and KV bandwidth while keeping many query heads.

### 3. Why does GQA reduce KV-cache memory?

Per token per layer, raw KV storage is approximately:

```text
2 × Hkv × Dh × bytes_per_element
```

The factor `2` is K and V. GQA reduces `Hkv` while keeping the query-head count high, so KV memory falls roughly in proportion to the KV-head reduction.

### 4. Why is RoPE applied to Q and K?

Attention similarity is computed from `QKᵀ`. RoPE rotates Q and K by position-dependent rotations. Because `R_mᵀR_n = R_(n-m)`, the resulting dot product naturally depends on relative position. V is not part of the similarity score, so standard RoPE does not need to rotate V.

### 5. RMSNorm vs LayerNorm?

LayerNorm subtracts the mean and normalizes by variance. RMSNorm removes the mean-centering step and scales by the root-mean-square magnitude. Systems-wise, both are reduction + elementwise operations rather than large GEMMs; actual speed depends heavily on fusion and kernel implementation.

### 6. What is SwiGLU?

A common form is:

```text
SiLU(XW_gate) ⊙ (XW_up)
```

followed by a down projection. It uses two expansion projections—gate and up—then elementwise gating before contracting back to hidden size.

---

## Inference

### 7. Prefill vs decode?

Prefill processes all prompt tokens in parallel and creates the initial KV cache. It tends to expose large GEMMs and often high arithmetic intensity. Decode processes one new token per active sequence per step while reading model weights and historical KV. At small batches this often has much lower arithmetic intensity and can become bandwidth-limited.

### 8. Why is small-batch decode often memory-bandwidth-bound?

Each step has very few new token rows, so there is limited reuse of large model weights. The runtime also reads growing KV state. Compute per byte moved is therefore low compared with large-batch GEMMs. Increasing batch size amortizes weight reads across more output tokens and can move the workload toward a more compute-efficient regime.

### 9. Derive KV-cache memory.

For one request:

```text
KV bytes = 2 × layers × sequence_length × KV_heads × head_dim × bytes/element
```

Derive it from the shape of one layer's K and V tensors, then multiply across layers. For concurrency `B`, multiply by the active sequence/cache lengths appropriately.

### 10. Does KV cache make attention O(1)?

No. It avoids recomputing old K/V projections. A new query still attends over historical keys/values, so decode attention work and data traffic grow with context length.

### 11. Why does larger batch improve decode throughput?

The same model weights can be reused for many active sequences in one step. A `[B,D] @ W` operation becomes more GEMM-like as B increases, increasing arithmetic intensity and hardware utilization. The trade-off is more queueing/KV usage and potentially worse TPOT or tail latency.

### 12. What does prefix caching reuse?

It reuses the per-layer KV states for an identical cached prompt prefix. The uncached suffix still needs prefill. Prefix caching therefore saves prompt compute and can improve TTFT, but it introduces cache admission/eviction and cache-aware routing trade-offs.

### 13. Weight quantization vs KV quantization?

Weight quantization targets model footprint and weight bandwidth. KV quantization targets persistent per-request attention state and KV bandwidth. They solve different memory consumers and have different kernel/quality/dequantization trade-offs.

### 14. What is speculative decoding?

A cheaper draft mechanism proposes multiple candidate tokens; the target model verifies them and accepts a valid prefix. If acceptance is high enough, the system advances several output tokens per expensive target-model iteration. Speedup depends on draft cost, verification cost, acceptance rate, batch, and runtime integration.

---

## GPU performance

### 15. What is arithmetic intensity?

```text
Arithmetic intensity = FLOPs / bytes moved from the relevant slow memory level
```

Roofline bounds performance by the smaller of peak compute and `bandwidth × arithmetic intensity`. Low-AI workloads are typically bandwidth-limited; high-AI workloads can become compute-limited.

### 16. What is GPU occupancy?

Roughly, the fraction of the SM's maximum warps that are resident/active. It is constrained by registers, shared memory, blocks and thread limits. Occupancy helps latency hiding, but maximum occupancy is not the objective—an HBM-saturated kernel can already be optimal at lower occupancy.

### 17. What is coalesced memory access?

Threads in a warp access neighboring/aligned addresses so hardware can combine lane requests into a small number of efficient global-memory transactions. Strided/scattered accesses can require many transactions and waste bandwidth.

### 18. Why does shared-memory tiling help GEMM?

A tile loaded once from HBM can be reused for many multiply-accumulate operations by threads in a block. This increases arithmetic intensity and reduces expensive global-memory traffic.

### 19. Why is FlashAttention faster?

Naive attention materializes large score/probability intermediates in HBM. FlashAttention tiles Q/K/V, computes score blocks in on-chip memory, and uses online softmax so the full `S×S` matrix does not need to be written/read from HBM. The main insight is IO reduction, not approximate attention.

### 20. What do CUDA Graphs optimize?

Repeated kernel launches and CPU/control overhead. A captured graph can replay a stable GPU workload with much lower launch overhead. It does not directly solve HBM bandwidth, algorithmic FLOPs, or communication.

---

## Distributed parallelism

### 21. What is tensor parallelism?

TP shards tensors/weight dimensions within each layer across GPUs. Column-parallel linears produce independent output shards; row-parallel linears produce partial outputs that need reduction. TP reduces per-GPU model/compute but introduces frequent communication, so it benefits from fast intra-node links.

### 22. Derive the classic TP MLP.

For `Y = activation(XA)B`, column-split `A=[A1|A2]`, so each GPU independently computes `H_i=activation(XA_i)`. Row-split `B=[B1;B2]`, so each GPU computes a partial `H_iB_i`; the partial outputs are summed with an AllReduce/related reduction.

### 23. Why can TP scaling get worse with more GPUs?

Per-GPU compute falls, but communication frequency remains and message sizes/topology may become less efficient. When extra collective latency/bandwidth cost exceeds saved compute, end-to-end latency worsens.

### 24. TP vs PP?

TP shards each layer and usually communicates at many layer boundaries; PP shards model depth and communicates activations between stages. TP often prefers very fast intra-node links. PP can cross slower boundaries more naturally but introduces pipeline bubbles and stage-balance issues.

### 25. Why is AllToAll central to MoE expert parallelism?

Tokens originating on any rank may select experts owned by any other rank. Dispatch therefore sends different token subsets to different expert ranks, naturally forming an AllToAll pattern; outputs must then be returned/combined. Load imbalance can create stragglers even when average compute is low.

---

## Serving / runtime

### 26. What is continuous batching?

At generation-iteration boundaries, completed sequences leave and waiting sequences can enter, so the active batch changes dynamically. This avoids wasting slots when requests have different output lengths and is fundamental to high-throughput autoregressive serving.

### 27. What problem does PagedAttention solve?

Variable-length KV state causes allocation waste and fragmentation. Paged KV management stores cache in fixed-size blocks with logical-to-physical mappings so memory can be allocated incrementally and reused efficiently. Attention kernels use block tables to read non-contiguous KV. It primarily improves effective memory utilization/concurrency; it does not eliminate historical KV reads.

### 28. What is chunked prefill?

Long prompt prefill is split into smaller chunks so the scheduler can interleave it with decode work. This reduces head-of-line blocking and protects active-request TPOT, at the cost of more scheduling steps and potentially longer completion time for one long prefill.

### 29. What is prefill/decode disaggregation?

Prefill and decode run on separate worker pools because their resource characteristics differ. This can reduce interference and allow separate scaling, but now KV state must be transferred and the two stages must be load-balanced. KV transfer/network and stage queues become new bottlenecks.

### 30. How do you debug low LLM-serving throughput?

First define the workload and exact metric. Split queueing from service time, inspect runtime metrics such as batch size, scheduled tokens, KV pressure and preemption, then use Nsight Systems to identify CPU gaps, underfilled kernels, communication, memcpy or synchronization. Only after locating hot kernels should you use Nsight Compute. Optimize one evidence-backed bottleneck and remeasure latency, throughput and memory regressions.

---

## How to use this page

For each answer:

1. Say it aloud in under 60 seconds.
2. Change one assumption or number.
3. Derive the consequence without looking up notes.
4. Follow with one deeper question from [`100-questions.md`](100-questions.md).

If the wording is memorized but the derivation breaks when the interviewer changes the numbers, the topic is not learned yet.
