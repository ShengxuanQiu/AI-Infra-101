# 04 — Distributed Parallelism

## Goal

Understand parallelism as **tensor placement + communication**, not as a list of acronyms.

## Collectives you must know

- Broadcast
- Reduce
- AllReduce
- AllGather
- ReduceScatter
- AllToAll
- point-to-point Send/Recv

Know what each rank owns before and after the collective and what communication pattern it enables.

## Parallelism strategies

### Data Parallelism (DP)

Replicate model computation; shard the batch. Gradients need synchronization in training. In inference, data-parallel replicas commonly serve different requests.

### Tensor Parallelism (TP)

Shard weight/tensor dimensions *within* a layer. The classic Megatron pattern combines column- and row-parallel linear layers to keep large intermediate activations sharded and synchronize at selected boundaries.

For an MLP:

```text
X → [A1 | A2]      # column split of first projection
   → activation independently
   → [B1 ; B2]     # row split of second projection
   → partial outputs summed (e.g. AllReduce)
```

You should be able to derive this with matrix algebra.

### Pipeline Parallelism (PP)

Shard model depth. Learn microbatches, bubbles, 1F1B intuition, and why pipeline communication is point-to-point activation traffic rather than frequent full-tensor collectives.

### Expert Parallelism (EP)

Shard experts in an MoE layer. Tokens must be routed to expert-owning ranks and returned. This naturally creates AllToAll-style communication and load-balancing challenges.

### Sequence / Context Parallelism

Know the high-level distinction: sequence dimensions/activations are partitioned, but attention requires access to information spanning sequence partitions, which creates communication around attention.

## Interconnect hierarchy

Know why topology matters:

```text
HBM ↔ GPU
GPU ↔ GPU: NVLink / NVSwitch (when available)
GPU ↔ host / devices: PCIe
node ↔ node: InfiniBand / RoCE / Ethernet depending system
```

The exact bandwidth changes by generation; the reasoning skill is to place each collective on the topology and ask whether communication can overlap with compute.

## Recommended resources

- **Stas Bekman — model parallelism notes** — clear Megatron tensor-parallel derivation:  
  https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism
- **Megatron-LM** — canonical implementation/docs for TP/PP/EP/CP and communication overlap:  
  https://github.com/NVIDIA/Megatron-LM
- **Hugging Face Tensor Parallelism docs** — concise row/column parallel illustrations:  
  https://github.com/huggingface/transformers/blob/main/docs/source/en/tensor_parallelism.md
- **GPU MODE lectures** — includes NCCL-oriented material:  
  https://github.com/gpu-mode/lectures
- **NCCL repository/docs**:  
  https://github.com/NVIDIA/nccl

## Interview prompts

- Derive a 2-way tensor-parallel MLP.
- Where are the communication boundaries in attention and MLP?
- Why is TP attractive inside a node with high-bandwidth links?
- Why can PP be attractive across slower links despite pipeline bubbles?
- What does ReduceScatter + AllGather have to do with AllReduce?
- Why is AllToAll central to expert parallelism?
- How do load imbalance and expert capacity affect MoE performance?
- When does adding GPUs make latency worse?
- What can compute/communication overlap hide, and what dependency prevents overlap?

See [`../cheatsheets/collectives.md`](../cheatsheets/collectives.md) for a quick reference.
