# 04 — Distributed Parallelism

> **Goal:** understand distributed LLM execution as **tensor placement + communication + topology**, not as a list of acronyms.

Whenever you hear TP/PP/DP/EP, ask:

1. What tensor/model state is sharded?
2. What does each rank own before compute?
3. What does each rank need after compute?
4. Which collective moves that data?
5. Does communication sit on NVLink/NVSwitch, PCIe, or the network?
6. Can it overlap with useful compute?

---

## 1. The basic collectives

Assume four ranks.

### Broadcast

One rank owns data; everyone receives it.

```text
before: [X] [ ] [ ] [ ]
after:  [X] [X] [X] [X]
```

### Reduce

Combine values onto one destination.

```text
before: [A] [B] [C] [D]
after:  [A+B+C+D] [ ] [ ] [ ]
```

### AllReduce

Everyone receives the reduction result.

```text
before: [A] [B] [C] [D]
after:  [Σ] [Σ] [Σ] [Σ]
```

This is central to replicated-data training and row-parallel reductions.

### AllGather

Each rank starts with one shard; everyone receives the full concatenation.

```text
before: [A] [B] [C] [D]
after:  [ABCD] [ABCD] [ABCD] [ABCD]
```

### ReduceScatter

Reduce values and keep only one reduced shard per rank.

```text
reduce + shard result
```

Conceptually:

```text
AllReduce ≈ ReduceScatter + AllGather
```

Many optimized implementations exploit this structure.

### AllToAll

Every rank sends a different shard to every other rank.

```text
rank i → distinct chunks for rank 0,1,2,3
```

This is especially important for MoE expert dispatch.

### Send / Recv

Point-to-point transfer. Pipeline parallelism commonly sends activations to the next stage rather than involving every rank in a collective.

See [`../cheatsheets/collectives.md`](../cheatsheets/collectives.md).

---

## 2. Latency and bandwidth model

A standard mental model for communication time is:

```text
T_comm ≈ α + bytes / BW
```

where:

- `α` = startup/latency term;
- `BW` = effective link bandwidth.

For multiple communication steps, collective algorithms add topology-dependent factors.

You do not need to memorize every formula for every NCCL algorithm, but you should understand two regimes:

```text
small message → latency/startup matters
large message → bandwidth dominates
```

This explains why fusing tiny collectives or overlapping large ones can matter.

---

## 3. Data Parallelism (DP)

Each rank has a model replica; data is sharded.

```text
GPU0: model copy + batch shard 0
GPU1: model copy + batch shard 1
...
```

### Training

Each rank computes local gradients; gradients must be synchronized, usually through AllReduce or ReduceScatter-based schemes.

### Inference

Replicas can serve different requests independently:

```text
request stream
   ↓ router
GPU replica 0
GPU replica 1
GPU replica 2
```

This scales throughput when a full model replica fits on each serving group.

### DP limitation

DP does **not** make a single oversized model fit. If one model copy exceeds device memory, you need model parallelism such as TP/PP/EP.

---

## 4. Tensor Parallelism (TP): start from matrix algebra

Suppose:

```text
Y = XW
```

### Column-parallel linear

Split W by output columns:

```text
W = [W1 | W2]
```

Then:

```text
Y = [XW1 | XW2]
```

Each rank can compute its output shard independently:

```text
GPU0: Y1 = XW1
GPU1: Y2 = XW2
```

No reduction is required immediately if the next operation can consume sharded outputs.

### Row-parallel linear

Split W by input rows:

```text
W = [W1
     W2]
```

and split X correspondingly:

```text
X = [X1 | X2]
```

Then:

```text
Y = X1W1 + X2W2
```

Each rank produces a partial output:

```text
GPU0: P0 = X1W1
GPU1: P1 = X2W2
```

and we need:

```text
Y = P0 + P1
```

which naturally maps to an AllReduce (or ReduceScatter if the next representation should remain sharded).

---

## 5. Worked TP example: MLP

Consider:

```text
Y = activation(XA) B
```

Use two GPUs.

### First projection: column split

```text
A = [A1 | A2]
```

Each GPU computes:

```text
GPU0: H1 = activation(X A1)
GPU1: H2 = activation(X A2)
```

The activation is elementwise, so it can run locally.

### Second projection: row split

```text
B = [B1
     B2]
```

Then:

```text
Y = H1B1 + H2B2
```

Each GPU computes a partial output and the partials are reduced.

This is the canonical Megatron-style idea:

```text
column parallel → local nonlinear op → row parallel → reduction
```

### Why this decomposition is elegant

If we gathered the expanded hidden activation after the first projection, communication would be larger and earlier. Keeping it sharded until the contraction reduces communication pressure.

---

## 6. Tensor-parallel attention

A simplified attention projection is:

```text
Q = XWq
K = XWk
V = XWv
```

Heads can be partitioned across TP ranks.

For example with 32 query heads and TP=4:

```text
GPU0: 8 query heads
GPU1: 8 query heads
GPU2: 8 query heads
GPU3: 8 query heads
```

Each rank performs attention on local heads.

The output projection then acts like a row-parallel linear whose partial outputs must be combined.

### GQA subtlety

If `Hkv` is small, TP degree must interact sensibly with KV-head placement. Some implementations replicate KV heads or constrain sharding depending on divisibility and kernel design.

The interview point: **parallelism must respect the model's tensor dimensions; acronyms do not automatically compose.**

---

## 7. Why TP prefers fast links

TP introduces communication inside nearly every Transformer layer.

So latency accumulates:

```text
layer 1 compute → collective
layer 2 compute → collective
layer 3 compute → collective
...
```

If the collective crosses a slow inter-node network, per-layer synchronization can become expensive.

Therefore a common topology-aware principle is:

> Put latency-sensitive, frequent TP collectives on high-bandwidth intra-node links when possible.

This is why NVLink/NVSwitch topology matters.

---

## 8. Pipeline Parallelism (PP)

PP shards layers by depth:

```text
GPU0: layers 0–19
GPU1: layers 20–39
GPU2: layers 40–59
GPU3: layers 60–79
```

A request flows through stages:

```text
stage 0 → stage 1 → stage 2 → stage 3
```

Communication is mostly activation Send/Recv between neighboring stages.

### Pipeline bubble

If a whole batch moves stage by stage, some stages sit idle.

Microbatching fills the pipeline:

```text
time →
S0: m0 m1 m2 m3
S1:    m0 m1 m2 m3
S2:       m0 m1 m2 m3
```

There are still warm-up/drain bubbles.

Training introduces forward/backward scheduling such as 1F1B. In pure inference, the scheduling problem differs, but the core concepts—stage balance, bubble, activation traffic—remain.

### PP advantage

Communication occurs at stage boundaries rather than after many layers' internal tensor operations, making PP potentially more tolerant of slower links than fine-grained TP.

### PP challenge

If stages have unequal execution time:

```text
stage 0: 3 ms
stage 1: 9 ms
stage 2: 4 ms
```

stage 1 becomes the throughput bottleneck.

---

## 9. Expert Parallelism (EP)

In MoE, only selected experts execute per token.

Suppose experts are distributed:

```text
GPU0: experts 0,1
GPU1: experts 2,3
GPU2: experts 4,5
GPU3: experts 6,7
```

A router may send tokens from any input rank to any expert rank:

```text
tokens
 ↓ router
AllToAll dispatch
 ↓
local experts
 ↓
AllToAll combine
 ↓
original token order
```

This creates three major systems problems:

1. **communication volume**;
2. **expert load imbalance**;
3. **dynamic token counts / buffer management**.

### Why MoE can be communication-heavy

MoE saves expert compute by activating only a subset of experts, but tokens still carry hidden vectors across devices.

So the workload can become:

```text
compute sparse
but
network / NVLink intensive
```

### Load imbalance

If one expert receives many more tokens:

```text
expert 0: 100 tokens
expert 1:  12 tokens
expert 2:   8 tokens
```

execution may wait for the overloaded expert.

Average FLOPs alone will not explain latency.

---

## 10. Context / sequence parallelism

Different systems use related terms differently, so focus on the principle.

A long sequence is partitioned across ranks:

```text
rank 0: tokens 0 ... S/2-1
rank 1: tokens S/2 ... S-1
```

But attention couples queries to keys/values across the sequence.

Therefore ranks must exchange some attention-related state or participate in communication patterns that make remote context visible.

Use this family of techniques when sequence-related activation/KV memory or long-context computation is the limiting dimension.

---

## 11. Interconnect hierarchy

A simplified topology:

```text
HBM ↔ GPU SMs
GPU ↔ GPU: NVLink / NVSwitch when available
GPU ↔ CPU/device: PCIe
node ↔ node: InfiniBand / RoCE / Ethernet depending cluster
```

The exact values vary by GPU generation and deployment.

For every collective ask:

```text
Where does this data physically travel?
```

A topology-oblivious parallel plan can look good on paper and perform poorly in practice.

---

## 12. Compute/communication overlap

Suppose a layer has independent compute and an AllReduce.

If the next useful compute does not depend on the full reduction result, communication may overlap:

```text
compute A ───────────
       communication ─────────
              compute B ───────
```

Effective time can approach:

```text
T ≈ max(T_compute, T_comm)
```

rather than:

```text
T = T_compute + T_comm
```

But dependencies limit overlap.

The interview question is not merely “can NCCL run asynchronously?” It is:

> **Is there independent useful work available while communication is in flight?**

---

## 13. Choosing TP vs PP vs DP vs EP

### Use DP when

- a serving replica fits;
- you want throughput across independent requests;
- communication between replicas is unnecessary on inference critical path.

### Use TP when

- a layer/model needs multiple GPUs;
- fast intra-node links are available;
- latency from frequent collectives is acceptable.

### Use PP when

- model depth can be partitioned cleanly;
- TP group size should remain smaller;
- topology favors stage communication over per-layer collectives.

### Use EP when

- the model is MoE;
- experts need to be sharded;
- AllToAll and load balancing are manageable.

Large deployments combine them:

```text
DP × TP × PP × EP × CP
```

But every added dimension increases scheduling and topology complexity.

---

## 14. Worked design example

Suppose a model cannot fit on one GPU but fits across 8 GPUs in one NVSwitch node.

A reasonable first option is TP=8 because:

- all GPUs are connected by a fast intra-node fabric;
- model layers can be sharded;
- no inter-node hop is required.

Now suppose serving grows to 32 GPUs across four 8-GPU nodes.

One common design space is:

```text
TP=8 inside each node
DP=4 across nodes
```

Each node is a model-parallel replica serving independent traffic.

Alternative PP/TP combinations may be better depending on model size, latency, link topology, and memory.

The point is not that `TP=8, DP=4` is always correct. The point is that **parallelism maps onto physical topology and workload objectives.**

---

## 15. Why adding GPUs can make latency worse

Possible reasons:

- collective overhead exceeds compute saved;
- inter-node links are slower;
- messages become too small to use bandwidth efficiently;
- synchronization increases;
- imbalance creates stragglers;
- CPU/runtime overhead grows;
- TP degree interacts poorly with head/expert dimensions.

Scaling efficiency is empirical, not guaranteed.

---

## 16. Common traps

### “AllReduce is just copying tensors.”

It combines values and distributes the reduced result.

### “TP always reduces latency because each GPU computes less.”

It also adds communication and synchronization.

### “PP has no communication.”

It sends activations between stages and suffers bubbles/imbalance.

### “MoE is always faster because fewer experts run.”

Routing, AllToAll, imbalance, and small expert GEMMs can dominate.

### “Network bandwidth is the only communication metric.”

Latency/startup matters, especially for small frequent messages.

---

## 17. Interview questions

1. Explain Broadcast/AllReduce/AllGather/ReduceScatter/AllToAll.
2. Why is AllReduce often decomposed conceptually into ReduceScatter + AllGather?
3. Derive a two-way column-parallel linear.
4. Derive a two-way row-parallel linear.
5. Derive the classic TP MLP.
6. Where does TP communicate in attention?
7. Why does TP prefer fast intra-node links?
8. What is a PP bubble?
9. Why can stage imbalance dominate PP throughput?
10. Why is AllToAll natural for MoE?
11. What does expert load imbalance do to latency?
12. When can communication overlap with compute?
13. Why can adding GPUs slow inference?
14. How would you map 32 GPUs arranged as four 8-GPU NVSwitch nodes?
15. What changes if your TP collective crosses nodes?

---

## Short resources

- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Megatron Core parallelism guide: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- Stas Bekman model-parallel notes: https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism
- Hugging Face TP docs: https://huggingface.co/docs/transformers/perf_infer_gpu_multi
- NCCL: https://github.com/NVIDIA/nccl
- GPU MODE lectures: https://github.com/gpu-mode/lectures

---

## Definition of done

Given a Transformer and a GPU topology, you can:

- derive row/column TP from matrix algebra;
- identify communication boundaries;
- explain PP bubbles and stage balance;
- explain MoE AllToAll and load imbalance;
- reason about DP replicas vs model parallelism;
- state where each collective travels physically;
- discuss whether communication can overlap with compute.

**Next:** [05 — LLM Serving](05-llm-serving.md)
