# 100 AI Infra Interview Questions

Use these as oral drills. A good answer usually includes **mechanism + quantitative intuition + trade-off + how you would verify it**.

## Transformer & model execution

1. Walk through one decoder-only Transformer block with tensor shapes.
2. Why is attention scaled by 1/sqrt(head_dim)?
3. What does a causal mask do and where is it applied?
4. Compare MHA, MQA and GQA.
5. Why does GQA reduce KV-cache memory?
6. What does RoPE encode and why is it applied to Q/K?
7. RMSNorm vs LayerNorm: what computation differs?
8. Write the SwiGLU computation and identify its projections.
9. What are the dominant GEMMs in one decoder block?
10. How does an MoE router choose experts?
11. Why can MoE reduce active FLOPs without reducing total parameter memory proportionally?
12. What causes expert load imbalance?
13. What does top-k routing change when k=1 vs k=2?
14. Which operations in a Transformer are reductions, GEMMs, elementwise or attention?
15. How would you estimate parameters in the attention and MLP parts of a block?

## Inference & KV cache

16. What exactly happens in prefill?
17. What exactly happens in one decode step?
18. Why is small-batch decode often bandwidth-bound?
19. Why can prefill reach higher arithmetic intensity?
20. Derive KV-cache bytes from model dimensions.
21. How does context length affect KV memory?
22. Why does the KV cache avoid recomputing historical K/V?
23. What computation still grows with context even with a KV cache?
24. How do MQA/GQA change the cache formula?
25. What is prefix caching and what is reused?
26. When would prefix caching have poor hit rate?
27. What is KV-cache quantization?
28. What are the risks of offloading KV cache to host memory?
29. Why can larger batches improve decode throughput?
30. Why can larger batches hurt latency?
31. Explain top-k vs top-p sampling.
32. Explain speculative decoding in draft/verify/accept terms.
33. What controls speculative-decoding speedup?
34. Why can a weak draft model hurt speculative decoding?
35. What changes in inference memory after weight quantization?

## GPU & kernels

36. What is a warp?
37. What is SIMT?
38. What happens under warp divergence?
39. What is an SM?
40. Registers vs shared memory vs L2 vs HBM?
41. What is global-memory coalescing?
42. What is a shared-memory bank conflict?
43. What is occupancy and why is higher not always better?
44. How does the GPU hide memory latency?
45. Define arithmetic intensity.
46. Explain the Roofline model.
47. Why is GEMM usually easier to make compute-efficient than GEMV?
48. What is kernel fusion and why can it help?
49. What is kernel-launch overhead?
50. What are CUDA streams used for?
51. What problem can CUDA Graphs reduce?
52. Explain FlashAttention as an IO optimization.
53. Why does tiling improve locality?
54. How can register pressure reduce occupancy?
55. What evidence distinguishes compute saturation from memory saturation?

## Distributed parallelism

56. Explain AllReduce.
57. Explain AllGather.
58. Explain ReduceScatter.
59. Explain AllToAll.
60. How are ReduceScatter + AllGather related to AllReduce?
61. Derive a 2-way column-parallel linear layer.
62. Derive a 2-way row-parallel linear layer.
63. Derive the Megatron tensor-parallel MLP.
64. Where does tensor-parallel attention communicate?
65. Why does TP prefer high-bandwidth intra-node links?
66. What is pipeline parallelism?
67. What creates a pipeline bubble?
68. What is 1F1B at a high level?
69. What does expert parallelism shard?
70. Why does EP induce AllToAll traffic?
71. What is sequence/context parallelism?
72. What is compute/communication overlap?
73. What dependency can prevent overlap?
74. Why can adding GPUs reduce scaling efficiency?
75. How would you choose TP vs PP for a model that barely fits?

## Serving & runtime

76. Define TTFT, TPOT/ITL, E2E latency and throughput.
77. What is goodput?
78. Why is P99 important for serving?
79. Static batching vs continuous batching?
80. What inefficiency does PagedAttention target?
81. Why does PagedAttention mainly help concurrency rather than single-request arithmetic?
82. What is chunked prefill?
83. How can a long prefill hurt decode TPOT?
84. Why might a scheduler preempt a request?
85. Recompute vs swap: what is the trade-off?
86. What state must a serving scheduler track?
87. How does prefix-cache-aware scheduling differ from FCFS?
88. What is prefill/decode disaggregation?
89. What new cost appears after P/D disaggregation?
90. How would you trace one request through vLLM?
91. What is the role of a KV-cache manager?
92. What is a model runner?
93. What does an attention backend abstract?
94. How does SGLang RadixAttention relate to prefix reuse?
95. Why can CPU scheduler overhead matter when GPU decode becomes very fast?

## Profiling & system design

96. What do you inspect first when throughput is lower than expected?
97. Nsight Systems vs Nsight Compute?
98. Why is nvidia-smi utilization insufficient for diagnosis?
99. How do you identify CPU-GPU bubbles?
100. What does many tiny kernels look like on a timeline?
101. How do you verify an optimization did not just change the workload?
102. What workload dimensions must accompany an LLM benchmark?
103. How would you diagnose high TTFT but normal TPOT?
104. How would you diagnose normal TTFT but bad TPOT?
105. How would you design serving for a 70B model with strict P99 TPOT?
106. What changes if requests share a huge system prompt?
107. What changes if contexts are 100K tokens?
108. What changes for MoE serving?
109. What trade-off would make you choose P/D disaggregation?
110. How do you explain a project optimization from evidence rather than anecdotes?
