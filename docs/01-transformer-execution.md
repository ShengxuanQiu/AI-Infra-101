# 01 — Transformer Execution

> **Goal:** see a Transformer block as an execution graph: tensor shapes, GEMMs, reductions, memory traffic, cache state, and natural parallelization boundaries.

If you can only recite the attention equation, you do not yet have the systems view. For AI Infra, every operator should trigger four questions:

1. **What are the input/output shapes?**
2. **What computation happens?**
3. **What data must move?**
4. **What state or communication does this create?**

---

## 1. The decoder block at a glance

A modern Llama/Qwen-style decoder block is roughly:

```text
x
│
├─ RMSNorm
│
├─ Q/K/V projections
│
├─ RoPE(Q, K)
│
├─ causal attention (MHA / GQA)
│
├─ output projection
│
└─ residual add ───────────────┐
                               │
                               ▼
                              x'
                               │
                               ├─ RMSNorm
                               ├─ SwiGLU MLP
                               └─ residual add
                                      │
                                      ▼
                                   output
```

With MoE, the dense MLP is replaced by:

```text
hidden states
    ↓
router logits
    ↓
top-k expert selection
    ↓
token dispatch
    ↓
expert MLPs
    ↓
weighted combine
```

The interesting systems boundary is that dense attention/MLP are mostly regular matrix operations, while MoE introduces **dynamic routing, imbalance, and communication**.

---

## 2. Shapes you should derive without notes

Use:

- `B`: batch size
- `S`: sequence length
- `D`: hidden size
- `Hq`: number of query heads
- `Hkv`: number of KV heads
- `Dh`: head dimension, usually `D / Hq`

The hidden state is:

```text
X: [B, S, D]
```

After projections:

```text
Q: [B, Hq,  S, Dh]
K: [B, Hkv, S, Dh]
V: [B, Hkv, S, Dh]
```

For MHA, `Hq = Hkv`.

For MQA, `Hkv = 1`.

For GQA:

```text
1 < Hkv < Hq
```

Each KV head is shared by:

```text
Hq / Hkv
```

query heads.

### Example: 32 query heads, 8 KV heads

Each KV head serves four query heads:

```text
Q heads:  0 1 2 3 | 4 5 6 7 | ...
KV head:      0    |     1   | ...
```

Conceptually, K/V can be repeated to match Q heads:

```python
repeat = Hq // Hkv
K_for_attention = repeat_interleave(K, repeat, head_axis)
V_for_attention = repeat_interleave(V, repeat, head_axis)
```

A good implementation may avoid physically repeating K/V and instead encode this mapping in the attention kernel.

---

## 3. Attention: math → execution

Scaled dot-product attention:

```text
Attention(Q,K,V) = softmax(QKᵀ / √Dh + mask)V
```

For one head in prefill:

```text
Q: [S, Dh]
K: [S, Dh]
V: [S, Dh]

QKᵀ: [S, S]
P V: [S, Dh]
```

The conceptual attention matrix is quadratic in sequence length:

```text
[S, S]
```

### Why divide by `sqrt(Dh)`?

Suppose entries of Q and K are roughly zero-mean with unit variance. A dot product sums `Dh` terms, so its variance grows approximately with `Dh`.

Without scaling:

```text
Var(q · k) ∝ Dh
```

Large logits push softmax into saturated regions. Dividing by `sqrt(Dh)` approximately keeps the scale stable as head dimension changes.

A strong interview answer is not just “because the paper says so”; it connects dot-product variance to softmax conditioning.

---

## 4. Causal masking

Decoder-only generation must not allow token `i` to attend to future token `j > i`.

Conceptually:

```text
score[i, j] = -∞  if j > i
```

Then:

```text
softmax(score)[i, j] ≈ 0
```

Pseudocode:

```python
scores = q @ k.transpose(-1, -2) / sqrt(head_dim)
mask = upper_triangular_matrix(fill=-inf)
scores = scores + mask
probs = softmax(scores, dim=-1)
out = probs @ v
```

During single-token decode, the new query sits at the end of the prefix, so every cached key is in its past; a full triangular mask is usually unnecessary for that one query.

---

## 5. Numerically stable softmax

The naive form:

```text
exp(x_i) / Σ exp(x_j)
```

can overflow for large logits.

Use:

```text
m = max(x)
softmax(x_i) = exp(x_i - m) / Σ exp(x_j - m)
```

Pseudocode:

```python
def stable_softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    z = exp(x - m)
    return z / z.sum(dim=dim, keepdim=True)
```

This is a recurring systems pattern: a mathematically simple reduction becomes a **max reduction + exponentiation + sum reduction + normalization**, which matters for kernel design.

---

## 6. RMSNorm

RMSNorm removes the mean-centering step used by LayerNorm.

For a hidden vector `x ∈ R^D`:

```text
rms(x) = sqrt(mean(x²) + eps)
RMSNorm(x) = x / rms(x) ⊙ weight
```

Pseudocode:

```python
def rmsnorm(x, weight, eps=1e-6):
    rms = sqrt(mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight
```

Systems view:

```text
read x
→ square
→ reduction over D
→ rsqrt
→ elementwise scale
→ write output
```

This is much more bandwidth/reduction-shaped than GEMM-shaped, which is why fusing normalization with neighboring operations can matter.

### RMSNorm vs LayerNorm

LayerNorm:

```text
(x - mean(x)) / sqrt(var(x) + eps)
```

RMSNorm:

```text
x / sqrt(mean(x²) + eps)
```

You should know the structural difference; you do not need to claim a universal performance advantage because real performance depends on kernels and fusion.

---

## 7. RoPE: the useful intuition

Rotary Position Embedding applies a position-dependent rotation to pairs of Q/K dimensions.

For a 2D pair:

```text
[x0']   [ cos θ  -sin θ ] [x0]
[x1'] = [ sin θ   cos θ ] [x1]
```

Different dimension pairs use different frequencies.

Why Q and K?

Attention depends on their dot product. Rotating Q at position `m` and K at position `n` gives:

```text
(R_m q)ᵀ(R_n k)
= qᵀ R_mᵀ R_n k
= qᵀ R_(n-m) k
```

The score naturally depends on **relative position `n-m`**.

That is the key interview insight.

V is not part of the similarity score, so standard RoPE is applied to Q/K rather than V.

### Minimal pseudocode

```python
def rotate_half(x):
    x1, x2 = split_even_pairs(x)
    return concat(-x2, x1)

q_rot = q * cos + rotate_half(q) * sin
k_rot = k * cos + rotate_half(k) * sin
```

Exact layouts vary across implementations.

---

## 8. MHA vs MQA vs GQA

### MHA

```text
Hq = Hkv
```

Every query head has its own K/V head.

Pros:
- maximum per-head flexibility.

Serving cost:
- largest KV cache among the three for fixed `Hq` and `Dh`.

### MQA

```text
Hkv = 1
```

All query heads share one K/V head.

Pros:
- very small KV cache and K/V bandwidth.

Trade-off:
- stronger sharing constraint.

### GQA

```text
1 < Hkv < Hq
```

A compromise between MHA and MQA.

The crucial serving formula is:

```text
KV bytes/token/layer = 2 × Hkv × Dh × bytes_per_element
```

So reducing `Hkv` directly reduces persistent KV state.

### Example

Assume:

```text
Hq  = 32
Hkv = 8
Dh  = 128
BF16 = 2 bytes
```

KV per token per layer:

```text
2 × 8 × 128 × 2
= 4096 bytes
= 4 KiB
```

With full MHA (`Hkv=32`):

```text
16 KiB/token/layer
```

GQA gives a 4× reduction in KV storage here.

---

## 9. SwiGLU MLP

A common gated MLP is:

```text
gate = SiLU(X W_gate)
up   = X W_up
h    = gate ⊙ up
out  = h W_down
```

Pseudocode:

```python
def swiglu_mlp(x):
    gate = silu(x @ W_gate)
    up = x @ W_up
    return (gate * up) @ W_down
```

Systems view:

```text
GEMM gate ─┐
           ├─ elementwise multiply → GEMM down
GEMM up   ─┘
```

The gate/up projections are often fused at implementation level because they consume the same input.

---

## 10. One block: where the big GEMMs are

Ignoring biases and details, one decoder layer contains:

### Attention projections

```text
Q = X Wq
K = X Wk
V = X Wv
```

Often implemented as one fused QKV projection or related fused layout.

### Attention output

```text
O = Attention(Q,K,V) Wo
```

### MLP

```text
G = X W_gate
U = X W_up
Y = (SiLU(G) ⊙ U) W_down
```

This naturally suggests parallelization boundaries:

- split Q/K/V projections;
- split output projection;
- split MLP expansion/contraction;
- shard experts in MoE.

That becomes important in [04 — Distributed Parallelism](04-distributed-parallelism.md).

---

## 11. Parameter and FLOP estimation

For a rough dense decoder layer with hidden size `D` and MLP intermediate size `F`:

Attention linear weights are approximately:

```text
Wq: D × D
Wk: D × (Hkv Dh)
Wv: D × (Hkv Dh)
Wo: D × D
```

For full MHA, `Hkv Dh = D`, giving roughly `4D²` attention projection parameters.

A SwiGLU MLP has three major matrices:

```text
W_gate: D × F
W_up:   D × F
W_down: F × D
```

≈ `3DF` parameters.

For a matrix multiply `(M×K)(K×N)`, a common dense FLOP estimate is:

```text
≈ 2MKN FLOPs
```

because each multiply-accumulate counts roughly as two FLOPs.

This estimate is sufficient for most interview back-of-the-envelope reasoning.

---

## 12. Hand-write bar

### Tier S — from memory

You should be able to write:

- stable softmax;
- RMSNorm;
- scaled dot-product attention;
- causal mask;
- multi-head reshape/merge;
- GQA mapping;
- RoPE application to Q/K;
- SwiGLU;
- KV-cache append/read path.

### Tier A — reconstruct comfortably

- full decoder block;
- top-k MoE router;
- simple sampling loop;
- TP-friendly split of attention/MLP.

Do not memorize framework-specific APIs. The test is whether you can reconstruct the mechanism from tensor algebra.

---

## 13. Worked exercise: trace one token through a block

Suppose:

```text
B = 2
S = 1024
D = 4096
Hq = 32
Hkv = 8
Dh = 128
F = 14336
```

You should be able to annotate:

```text
X          [2, 1024, 4096]
Q          [2, 32, 1024, 128]
K/V        [2,  8, 1024, 128]
score      [2, 32, 1024, 1024]   # conceptual
attn out   [2, 32, 1024, 128]
merged     [2, 1024, 4096]
MLP up     [2, 1024, 14336]
output     [2, 1024, 4096]
```

Then ask:

- Which tensors persist across decode? → K/V cache.
- Which conceptual tensor should a memory-efficient attention kernel avoid materializing? → the full score/probability matrix.
- Which operations look GEMM-like? → projections and MLP.
- Which look reduction/elementwise-like? → norms, softmax, activation.

That is exactly the systems perspective you want.

---

## 14. Common interview traps

### “Attention is O(S²), so inference must always be compute-bound.”

Wrong. Big-O does not tell you hardware bottlenecks. Decode with one query token has a very different shape from full-sequence prefill.

### “GQA makes the model hidden size smaller.”

Wrong. It reduces the number of **K/V heads**, not the number of query heads or hidden width.

### “FlashAttention approximates attention.”

Wrong. FlashAttention is an IO-aware exact attention algorithm (modulo ordinary floating-point behavior).

### “KV cache stores hidden states.”

Be precise: the cache stores the K/V states required by future attention, usually per layer.

---

## 15. Interview questions

You should answer these cleanly:

1. Why scale attention by `1/sqrt(Dh)`?
2. Why is RoPE applied to Q/K?
3. Derive MHA/MQA/GQA shapes.
4. Why does GQA reduce serving memory?
5. RMSNorm vs LayerNorm?
6. Write SwiGLU from memory.
7. Identify all major GEMMs in a decoder block.
8. What operations are reductions rather than GEMMs?
9. Where can TP split attention and MLP naturally?
10. Why is MoE compute-sparse but potentially communication-heavy?
11. Which attention intermediate causes quadratic storage in a naive implementation?
12. During decode, why can one query reuse cached K/V from previous tokens?

---

## Short resources

### Implementation practice

- Modern Transformer from scratch: https://github.com/ruisp666/transformer-from-scratch
- Llama from scratch: https://github.com/bkitano/llama-from-scratch
- TorchCode: https://github.com/duoan/TorchCode
- TorchLeet: https://github.com/Exorust/TorchLeet

### Systems-oriented references

- GPU MODE awesomeMLSys: https://github.com/gpu-mode/awesomeMLSys
- JAX Scaling Book: https://jax-ml.github.io/scaling-book/

---

## Definition of done

Given a Llama/Qwen-style config, you can:

- sketch one block;
- annotate every major tensor shape;
- derive MHA/GQA KV storage;
- identify GEMM vs reduction/elementwise work;
- estimate rough parameter/FLOP counts;
- write attention, RMSNorm, RoPE, GQA and SwiGLU without searching;
- explain which states survive into the next decode step.

**Next:** [02 — LLM Inference](02-llm-inference.md)
