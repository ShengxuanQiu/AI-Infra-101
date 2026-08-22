# 01 — Transformer Execution

## Goal

Do not learn a Transformer only as equations. Learn it as an **execution graph**: tensor shapes, matrix multiplications, memory traffic, cache state, and parallelization boundaries.

## Must know

### Decoder-only block

Be able to trace:

```text
hidden states
  → RMSNorm
  → Q/K/V projections
  → RoPE(Q, K)
  → causal MHA/GQA
  → output projection
  → residual
  → RMSNorm
  → SwiGLU MLP (gate/up/down)
  → residual
```

For MoE, replace the dense MLP with:

```text
router logits → top-k experts → token dispatch → expert MLP → combine
```

### Shapes

For batch `B`, sequence `S`, hidden dimension `D`, query heads `Hq`, KV heads `Hkv`, head dimension `Dh`:

```text
X        [B, S, D]
Q        [B, Hq,  S, Dh]
K, V     [B, Hkv, S, Dh]
score    [B, Hq,  S, S]   # conceptual dense attention
output   [B, Hq,  S, Dh]
```

For GQA, multiple query heads share one KV head. You must be able to explain the grouping and how this reduces KV-cache size.

### Hand-write bar

You should be able to implement or write clean pseudocode for:

- numerically stable softmax;
- RMSNorm;
- scaled dot-product attention;
- causal attention mask;
- multi-head reshape/merge;
- GQA head mapping/repetition;
- RoPE application to Q/K;
- SwiGLU MLP;
- a single decoder block;
- KV-cache append/read path;
- simple top-k router for MoE.

Do **not** memorize framework APIs. The goal is to reconstruct the operation from the math and tensor shapes.

## Questions you should answer

1. Why scale attention logits by `1/sqrt(Dh)`?
2. Why is RoPE applied to Q/K rather than V?
3. What is the difference between MHA, MQA and GQA?
4. Why does GQA reduce serving memory?
5. RMSNorm vs LayerNorm: what is removed and why might that matter?
6. Why does SwiGLU use gate/up/down projections?
7. What are the main GEMMs in one Transformer block?
8. Which operations are elementwise/reduction/attention/GEMM?
9. Where can tensor parallelism split the block naturally?
10. In MoE, what is compute-sparse but communication-heavy?

## Recommended short resources

### Best implementation references

- **Modern transformer from scratch** — RoPE, RMSNorm, SwiGLU, GQA, KV cache and MoE in one pedagogical repo:  
  https://github.com/ruisp666/transformer-from-scratch
- **Llama from scratch** — compact narrative implementation:  
  https://github.com/bkitano/llama-from-scratch
- **TorchCode** — auto-graded softmax/RMSNorm/SwiGLU/attention/GQA/RoPE/KV-cache exercises:  
  https://github.com/duoan/TorchCode
- **TorchLeet** — additional LLM implementation drills:  
  https://github.com/Exorust/TorchLeet

### Deeper reference

- GPU MODE awesomeMLSys — attention/FlashAttention/GQA reading list:  
  https://github.com/gpu-mode/awesomeMLSys

## Definition of done

Given a Llama/Qwen-style config, you can sketch one block, annotate every major tensor shape, identify the dominant matrix multiplications, estimate KV-cache size, and write GQA + KV-cache pseudocode without searching the web.
