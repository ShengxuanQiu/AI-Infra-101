# Hand-write / Implementation Checklist

The target is not memorizing framework calls. You should be able to reconstruct these from definitions.

## Tier S — do from memory

- [ ] stable softmax
- [ ] cross entropy from logits (using log-sum-exp intuition)
- [ ] RMSNorm
- [ ] scaled dot-product attention
- [ ] causal self-attention
- [ ] multi-head attention reshape/merge
- [ ] grouped-query attention
- [ ] RoPE application
- [ ] SwiGLU MLP
- [ ] KV-cache append + decode attention pseudocode
- [ ] top-k sampling
- [ ] top-p sampling

Recommended judge: https://github.com/duoan/TorchCode

## Tier A — strongly recommended

- [ ] Transformer decoder block
- [ ] simple MoE router + top-k dispatch pseudocode
- [ ] LRU cache
- [ ] producer-consumer queue with mutex/condition variable
- [ ] thread-safe bounded queue
- [ ] ring buffer
- [ ] matrix multiplication baseline
- [ ] reduction
- [ ] softmax CUDA/Triton optimization sketch
- [ ] simple tensor-parallel MLP simulation

## Tier B — role dependent

Kernel/performance roles:

- [ ] CUDA reduction with shared memory / warp primitives
- [ ] tiled GEMM
- [ ] fused elementwise/reduction kernel
- [ ] online softmax
- [ ] FlashAttention tiling pseudocode
- [ ] Triton matmul/softmax

Serving/runtime roles:

- [ ] scheduler pseudocode for waiting/running queues
- [ ] continuous batching step
- [ ] simple block-based KV allocator
- [ ] prefix-cache lookup policy sketch
- [ ] token-budget admission algorithm

## How to practice

For each problem:

1. explain the math first;
2. annotate shapes;
3. write the simplest correct version;
4. state numerical edge cases;
5. state performance bottlenecks;
6. only then optimize.
