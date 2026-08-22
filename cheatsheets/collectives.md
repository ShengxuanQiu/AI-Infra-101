# Collectives & Parallelism Cheat Sheet

| Primitive | Before | After | Typical AI use |
|---|---|---|---|
| Broadcast | one rank owns tensor | all ranks own copy | distribute metadata/parameters |
| Reduce | all ranks own partial values | one rank owns reduction | aggregation |
| AllReduce | all ranks own partial values | all ranks own reduced result | gradient sync, TP partial-output sum |
| AllGather | each rank owns shard | all ranks own concatenated tensor | reconstruct sharded activations/weights |
| ReduceScatter | all ranks own values | each rank owns shard of reduced result | sharded gradient/activation workflows |
| AllToAll | each rank owns chunks for every peer | chunks redistributed by destination | MoE token dispatch/combine |
| Send/Recv | one peer | another peer | pipeline-stage activation transfer |

## Parallelism map

| Strategy | Shards | Primary benefit | Primary cost |
|---|---|---|---|
| DP | requests/batch | throughput / replicas | weight replication; training gradient sync |
| TP | tensor dimensions inside layer | fit + parallelize layer | frequent collectives; needs fast links |
| PP | model depth | fit large model; lower-frequency P2P | bubbles, stage imbalance |
| EP | experts | scale MoE expert capacity | AllToAll + load imbalance |
| SP/CP | sequence/activation dimensions | reduce activation/context footprint | communication around attention/normalization depending scheme |

## Megatron MLP pattern

```text
X replicated/sharded as required
  ↓
Column-parallel first linear: independent output shards
  ↓
activation on each shard
  ↓
Row-parallel second linear: partial hidden outputs
  ↓
AllReduce (or equivalent reduce-scatter sequence-parallel formulation)
```

Mental rule: **keep the largest intermediate activation sharded and synchronize at a smaller boundary when possible.**
