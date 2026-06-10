# Introduction Summary: Intro to GPUs and GPU Parallelism

---

## Notebook 02: Why GPUs for Deep Learning

### Counting Transformer FLOPs (`count_transformer_flops`)

**Core Formula:** For any matmul `(M, K) @ (K, N)` → FLOPs = `2 * M * K * N`

The "2" = K multiplications + K-1 additions per output element ≈ 2K ops × M×N elements.

**Config (LLaMA-2 7B):** d_model=4096, n_heads=32, d_head=128, d_ff=11008, seq_len=2048, batch=1

#### 1. QKV Projection (28.9%)
```
FLOPs = 2 * batch * seq_len * d_model * (3 * d_model)
```

**Shapes:**
- Input: `(batch, seq_len, d_model)` → `(1, 2048, 4096)`
- Weight: `(d_model, 3*d_model)` → `(4096, 12288)`
- Output: `(1, 2048, 12288)`

**M, K, N identification:** The matmul is effectively `(batch*seq_len, d_model) @ (d_model, 3*d_model)`:
- M = batch × seq_len = 2048 (number of token vectors)
- K = d_model = 4096 (input dimension — the shared/contracted dimension)
- N = 3 × d_model = 12288 (output dimension — Q, K, V concatenated)
- FLOPs = 2 × 2048 × 4096 × 12288 ≈ 206B

**Why 3×?** Fuses W_Q, W_K, W_V into one `(4096, 12288)` matmul — same total work as three separate `(4096, 4096)` projections, but one kernel launch instead of three → more GPU-efficient.

#### 2. Attention Scores — Q @ K^T (4.8%)
```
FLOPs = 2 * batch * n_heads * seq_len * d_head * seq_len
```

**Shapes (per head):**
- Q: `(seq_len, d_head)` → `(2048, 128)`
- K^T: `(d_head, seq_len)` → `(128, 2048)`
- Output: `(seq_len, seq_len)` → `(2048, 2048)` — the full attention matrix

**M, K, N identification:** `(seq_len, d_head) @ (d_head, seq_len)`:
- M = seq_len = 2048 (query positions)
- K = d_head = 128 (the dimension we dot-product over)
- N = seq_len = 2048 (key positions)
- Per head: 2 × 2048 × 128 × 2048 ≈ 1.07B FLOPs
- × 32 heads × 1 batch = ~34.4B total

**Why only 4.8%?** d_head is small (128). Compare to QKV's contracted dimension of d_model=4096 — that's 32× larger. The attention matmul contracts over d_head, not d_model.

**The quadratic warning:** This term has seq_len² in it. At seq_len=2048 it's cheap, but at 8192 it would be 16× more expensive, and eventually overtakes the linear projections.

#### 3. Attention Output — Attn @ V (4.8%)
```
FLOPs = 2 * batch * n_heads * seq_len * seq_len * d_head
```

**Shapes (per head):**
- Attn weights: `(seq_len, seq_len)` → `(2048, 2048)`
- V: `(seq_len, d_head)` → `(2048, 128)`
- Output: `(seq_len, d_head)` → `(2048, 128)`

**M, K, N identification:** `(seq_len, seq_len) @ (seq_len, d_head)`:
- M = seq_len = 2048 (output positions)
- K = seq_len = 2048 (positions we're summing over — the attention-weighted sum)
- N = d_head = 128 (value dimension)
- Per head: 2 × 2048 × 2048 × 128 ≈ 1.07B FLOPs

Same cost as Q@K^T — the dimensions are just transposed.

**Intuition:** For each of the 2048 output positions, we take a weighted average (weights from softmax) across all 2048 positions, producing a 128-dim vector. This is how each token "gathers" information from the tokens it decided to attend to.

#### 4. Output Projection (9.6%)
```
FLOPs = 2 * batch * seq_len * d_model * d_model
```

**Shapes:**
- Input: `(batch, seq_len, d_model)` → `(1, 2048, 4096)` — concatenated heads
- Weight: `(d_model, d_model)` → `(4096, 4096)`
- Output: `(1, 2048, 4096)`

**M, K, N identification:** `(batch*seq_len, d_model) @ (d_model, d_model)`:
- M = batch × seq_len = 2048
- K = d_model = 4096 (input from concatenated heads)
- N = d_model = 4096 (output)
- FLOPs = 2 × 2048 × 4096 × 4096 ≈ 68.7B

**Why it's needed:** Each head operated independently on its 128-dim slice. The output projection is the only place where heads can "talk to each other" — it learns which combinations of head outputs are useful.

**Why exactly 1/3 of QKV:** QKV projects `d_model → 3×d_model` (N=12288), output projects `d_model → d_model` (N=4096). Ratio is just the N dimension: 12288 vs 4096 = 3×.

#### 5. FFN Layer 1 (25.9%)
```
FLOPs = 2 * batch * seq_len * d_model * d_ff
```

**Shapes:**
- Input: `(batch, seq_len, d_model)` → `(1, 2048, 4096)`
- Weight: `(d_model, d_ff)` → `(4096, 11008)`
- Output: `(1, 2048, 11008)`

**M, K, N identification:** `(batch*seq_len, d_model) @ (d_model, d_ff)`:
- M = batch × seq_len = 2048
- K = d_model = 4096 (input dimension)
- N = d_ff = 11008 (expanded hidden dimension)
- FLOPs = 2 × 2048 × 4096 × 11008 ≈ 184.7B

**Why expand?** Attention mixes info *across positions* but stays within d_model dimensions. FFN expands to a wider space where element-wise non-linearity (GELU/SwiGLU) can "select" and transform features. Attention decides *what to look at*, FFN decides *what to do with it*.

**Why d_ff=11008 and not 4×4096=16384?** LLaMA uses SwiGLU activation, which has a *gate* projection in addition to the up-projection. The gate effectively halves usable width, so they compensate with `d_ff = (2/3) × 4 × d_model ≈ 11008` to keep total compute similar to a standard FFN.

**Note:** The non-linearity (GELU/SwiGLU) applied between FFN1 and FFN2 is element-wise — its FLOPs are negligible compared to the matmuls, which is why it's not counted here.

#### 6. FFN Layer 2 (25.9%)
```
FLOPs = 2 * batch * seq_len * d_ff * d_model
```

**Shapes:**
- Input: `(batch, seq_len, d_ff)` → `(1, 2048, 11008)`
- Weight: `(d_ff, d_model)` → `(11008, 4096)`
- Output: `(1, 2048, 4096)`

**M, K, N identification:** `(batch*seq_len, d_ff) @ (d_ff, d_model)`:
- M = batch × seq_len = 2048
- K = d_ff = 11008 (the wide dimension being contracted)
- N = d_model = 4096 (back to model dimension)
- FLOPs = 2 × 2048 × 11008 × 4096 ≈ 184.7B

**Why same cost as FFN Layer 1?** Dimensions are just swapped — Layer 1 does `(4096, 11008)`, Layer 2 does `(11008, 4096)`. In `2*M*K*N`, multiplication is commutative, so `d_model × d_ff` = `d_ff × d_model`. Up-projection and down-projection are always symmetric in FLOPs.

**The FFN "sandwich" intuition:**
```
d_model=4096 → [expand] → d_ff=11008 → [non-linearity] → d_ff=11008 → [compress] → d_model=4096
```
Without the non-linearity in between, two consecutive linear layers collapse into one (A @ B = C, just another linear map). The expansion gives the non-linearity more dimensions to "carve" decision boundaries in.

#### Key Insights
- **Linear projections = ~90% of FLOPs.** Attention scores (Q@K^T + Attn@V) = only ~10%. The "signature" operation of transformers is not the expensive part.
- **FFN is nearly half the compute (52%)** — often overlooked because attention gets all the press.
- **At seq_len=2048, the d_model² terms dominate.** The linear projections scale with d_model² (e.g., 4096²=16M per output element). Attention scales with seq_len² × d_head (2048² × 128). At seq_len=8192+, attention overtakes.
- **One layer: 713 GFLOPs. 32 layers forward: 22.81 TFLOPs. Forward + backward (3× forward): 68.44 TFLOPs/step.**
- **The 3× rule for training:** backward pass ≈ 2× forward (gradients for both weights AND inputs), so total training step ≈ 3× forward FLOPs.

---

### Benchmarking Real Neural Network Layers (Part 2)

#### The Benchmarking Pattern

**Warmup:** Run one forward+backward pass before timing to avoid measuring:
- CUDA lazy initialization (context setup, driver init — can add 100ms+)
- Kernel compilation/caching (first call for a given shape/dtype is slower)
- Memory allocator first-use overhead (pool allocation from OS)

**The synchronize sandwich:**
```
synchronize()       ← flush prior work so it doesn't leak into measurement
start = timer
<GPU work>
synchronize()       ← wait for GPU to actually finish
elapsed = timer - start
```
Without this, you'd just measure how fast Python can *enqueue* work (microseconds), not how long the GPU takes to *execute* it (milliseconds). GPU operations are asynchronous — they return to Python immediately.

**Fresh input each trial:** Avoids autograd graph interference from previous iterations (stale `.grad` buffers, graph references).

**Median over trials:** Robust to outliers (OS scheduling, GC, CUDA context switches). Mean would be skewed by one bad sample.

#### The Five Layers Tested

| Layer | Transformer Role | Bound Type |
|-------|-----------------|------------|
| Linear (1024→1024) | Q/K/V, output projection | Compute-bound |
| Linear (1024→4096) | FFN expansion | Compute-bound |
| LayerNorm | Pre-attention/FFN normalization | Memory-bound |
| GELU | FFN non-linearity | Memory-bound |
| Softmax | Attention score normalization | Memory-bound |

**Compute-bound:** Bottleneck is arithmetic throughput. Data is reused heavily (each weight participates in many multiply-adds). High arithmetic intensity.

**Memory-bound:** Bottleneck is data transfer speed. Each element is loaded, a cheap op is done, and it's written back. ALUs sit idle waiting for memory. Low arithmetic intensity.

#### Results (H100, batch=32, seq=512, d=1024)

```
Linear (1024->1024):   88.3x speedup  (GPU: 0.75ms fwd, 1.55ms bwd)
Linear (1024->4096):   62.8x speedup  (GPU: 2.81ms fwd, 5.84ms bwd)
LayerNorm:            242.4x speedup  (GPU: 0.08ms fwd, 0.31ms bwd)
GELU:                 552.9x speedup  (GPU: 0.06ms fwd, 0.12ms bwd)
Softmax:              386.7x speedup  (GPU: 0.06ms fwd, 0.22ms bwd)
```

#### The Counterintuitive Result: Memory-bound ops get HIGHER speedup ratios

**Why?** It's about how bad CPUs are at element-wise ops on large tensors:
- Input tensor: `(32, 512, 1024)` = 16.7M elements
- CPU: 16.7M sequential-ish memory accesses with limited parallelism (8-16 cores)
- GPU: 16.7M elements spread across thousands of threads, reading from 3.35 TB/s HBM in parallel → finishes in ~0.06ms

Meanwhile, CPUs have optimized BLAS libraries (MKL, AVX-512) for matmul, so they aren't *terrible* at Linear layers — just 60-90× slower rather than 400-500× slower.

#### The Real Takeaway: Absolute Time Matters More Than Speedup Ratio

- Linear (1024→1024) forward: **0.75ms** — this is where wall-clock goes
- GELU forward: **0.06ms** — basically free on GPU

In a full transformer step, 90%+ of wall-clock is in linear layers. The element-wise ops (GELU, Softmax, LayerNorm) are negligible on GPU despite their impressive speedup ratios.

#### Why Backward ≈ 2× Forward for Linear Layers

- Forward: `y = x @ W^T + b` (one matmul)
- Backward computes TWO gradients:
  - `grad_input = grad_output @ W` (same cost as forward)
  - `grad_weight = x^T @ grad_output` (another matmul of similar size)
- Total: 2 matmuls ≈ 2× the forward pass
- Observed: 1.548ms / 0.751ms = 2.06× ✓

---

### Part 3: Profiling a Complete Training Step

**What's measured:** Time spent in forward pass, backward pass, and optimizer step separately.

**The profiling pattern:**
```
synchronize()  → flush prior GPU work
t0 = timer
forward pass + loss
synchronize()  → wait for GPU to finish
t1 = timer
backward pass
synchronize()
t2 = timer
optimizer step
synchronize()
t3 = timer
```

`torch.cuda.synchronize()` is critical — GPU ops are async. Without it, you measure "time to launch kernel" not "time to complete."

**Typical results:**
```
Forward:    ~30% of total time
Backward:   ~55% of total time  ← ~2× forward
Optimizer:  ~15% of total time
```

**Why backward ≈ 2× forward:** Forward does 1 matmul per layer. Backward computes TWO gradients:
- grad_input = grad_output @ W (needed to propagate backward)
- grad_weight = x^T @ grad_output (needed to update this layer's weights)

Two matmuls ≈ 2× forward cost.

---

### Part 4: Memory Analysis — Why Single GPUs Run Out

**The 5 memory costs of training (per parameter):**

| Component | Bytes per param | For 7B model |
|-----------|----------------|--------------|
| Weights (FP16) | 2 | 14 GB |
| Master weights (FP32) | 4 | 28 GB |
| Gradients (FP16) | 2 | 14 GB |
| Adam 1st moment (FP32) | 4 | 28 GB |
| Adam 2nd moment (FP32) | 4 | 28 GB |
| **Total per param** | **16 bytes** | **112 GB** |

That's 112 GB just for parameters + optimizer — before any activations!

**Activations (grows with batch × seq_len):**

```
Per layer:
  Input/residual: batch × seq_len × d_model × 2 × 4 bytes
  Attention scores: batch × n_heads × seq_len × seq_len × 2 bytes  ← QUADRATIC!
  FFN intermediate: batch × seq_len × d_ff × 2 bytes
```

The attention scores have seq_len² — this is why long sequences blow up memory.

**Key insight: Parameters are independent of batch/seq, activations are not.**

| Component | Depends on batch/seq? |
|-----------|----------------------|
| Weights | No — fixed shape |
| Gradients | No — same shape as weights |
| Optimizer states | No — same shape as weights |
| **Activations** | **Yes — grows with batch × seq_len** |

This is why you can run the same trained model with batch=1 or batch=64 — weights don't change, only activations do.

**Why parallelism is needed:**
- 7B model training: ~112 GB params + ~50 GB activations = exceeds A100's 80 GB
- 70B model training: ~1.1 TB params alone = needs 14+ GPUs just for weights

---

### Part 5: Mixed Precision Training

**The idea:** Use FP16 for most computation (2× faster, half the memory), keep FP32 for critical parts (prevent underflow).

**What actually saves memory:**

| Component | FP32 training | Mixed precision | Savings |
|-----------|--------------|-----------------|---------|
| Master weights | 4 bytes/param | 4 bytes/param | Same |
| FP16 weights copy | — | 2 bytes/param | Extra cost |
| Gradients | 4 bytes/param | 2 bytes/param | 2× less |
| Adam m, v | 8 bytes/param | 8 bytes/param | Same |
| **Activations** | **4 bytes/element** | **2 bytes/element** | **2× less!** |

The big win is **activations** (which can be 30-60 GB for large batch/seq) and gradients in FP16. Weights/optimizer states stay in FP32 either way.

**GradScaler — solving the underflow problem:**
1. Forward pass in FP16 (fast)
2. Multiply loss by large number (e.g., 1024) before backward — prevents tiny gradients from becoming zero in FP16
3. Backward pass in FP16 (gradients are scaled up, stay representable)
4. Divide gradients back down before optimizer step
5. Optimizer updates FP32 master weights (precise accumulation)

**Why not pure FP16?** FP16's smallest value ≈ 0.00006. Gradients are often smaller than this → underflow to zero → training breaks. The scaler keeps them in representable range.

**Results:** ~1.5-2× speedup + ~30-40% less memory (from activations + gradients being half-size). FP16 matmuls also run 2× faster on Tensor Cores.

---

---

## Notebook 03: 5D Training Parallelism Overview

### Assembly Line Analogy

Think of training a large model like building cars on an assembly line:
- **Pipeline parallelism** = the assembly line stations (frame → engine → paint → interior)
- **Tensor parallelism** = splitting one station's job among multiple workers (4 painters, each painting one side of the same car)
- **Data parallelism** = running multiple identical assembly lines, each building a different car, then sharing what they learned

---

### 1. Data Parallelism

**What's split:** Training data (each GPU gets different samples)

**How it works:**
1. Every GPU holds a complete copy of the model (identical weights)
2. The global batch is split into chunks — GPU 0 gets samples 0–7, GPU 1 gets 8–15, etc.
3. Each GPU independently runs forward + backward on its chunk
4. All GPUs average their gradients (all-reduce)
5. Every GPU applies the same averaged gradient → models stay in sync

**Why it works mathematically:** The mean of local gradients equals the gradient of the full batch. Each GPU computes a partial mean, averaging them gives the global mean.

**Communication:** One all-reduce per training step (to average gradients). Everything else is independent.

**Pros:** Simple, works with any model that fits on one GPU, near-linear speedup.

**Cons:** Every GPU must hold the entire model. Memory usage doesn't decrease — only training speed increases.

**When to use:** Your model fits on one GPU, you want to train faster with more data.

---

### 2. Tensor Parallelism

**What's split:** Individual weight matrices (each GPU holds a slice of the layer)

#### Column-Parallel (split output dimension)

Split W into row chunks — each GPU computes a subset of output features:

```
W shape: (512, 512), split across 4 GPUs:

GPU 0: W[0:128, :]     → (128, 512)
GPU 1: W[128:256, :]   → (128, 512)
GPU 2: W[256:384, :]   → (128, 512)
GPU 3: W[384:512, :]   → (128, 512)
```

Each GPU gets the FULL input x but produces only 1/4 of the output:
```
GPU 0: y_0 = x @ W_0^T  → (batch, 128)   ← output features 0-127
GPU 1: y_1 = x @ W_1^T  → (batch, 128)   ← output features 128-255
GPU 2: y_2 = x @ W_2^T  → (batch, 128)   ← output features 256-383
GPU 3: y_3 = x @ W_3^T  → (batch, 128)   ← output features 384-511
```

**Combine: concatenate (all-gather)** → (batch, 512)

#### Row-Parallel (split input dimension)

Split W into column chunks — each GPU gets a subset of input connections:

```
W shape: (512, 512), split across 4 GPUs:

GPU 0: W[:, 0:128]     → (512, 128)
GPU 1: W[:, 128:256]   → (512, 128)
GPU 2: W[:, 256:384]   → (512, 128)
GPU 3: W[:, 384:512]   → (512, 128)
```

Input x is ALSO split — each GPU gets the corresponding 128 features:
```
GPU 0: y_0 = x[:, 0:128]   @ W_0^T  → (batch, 512)   ← partial sum
GPU 1: y_1 = x[:, 128:256] @ W_1^T  → (batch, 512)   ← partial sum
GPU 2: y_2 = x[:, 256:384] @ W_2^T  → (batch, 512)   ← partial sum
GPU 3: y_3 = x[:, 384:512] @ W_3^T  → (batch, 512)   ← partial sum
```

**Combine: sum (all-reduce)** → y = y_0 + y_1 + y_2 + y_3

Why summing works: a dot product can be split into chunks and summed:
`x · w = (x[0:128] · w[0:128]) + (x[128:256] · w[128:256]) + ...`

#### Key Difference

| | Column-Parallel | Row-Parallel |
|---|---|---|
| Each GPU computes | Subset of output features | Partial sum of ALL outputs |
| Input x | Same x on all GPUs | Split across GPUs |
| Combine operation | **Concatenate** (all-gather) | **Sum** (all-reduce) |
| Output per GPU | (batch, d_out/N) | (batch, d_out) |

#### Megatron-LM Pattern: Pair Column then Row

Column-parallel followed by row-parallel avoids extra communication:

```
FFN Layer 1 (column-parallel): x → split output → each GPU has (batch, d_ff/N)
                                    ↓ no communication needed here ↓
FFN Layer 2 (row-parallel):    takes split input → sum → each GPU has (batch, d_model)
```

The output of column-parallel is *already split* — which is exactly the input format row-parallel needs. So no communication between the two layers. Only one all-reduce at the end.

This pattern gives only **one all-reduce per attention block** and **one per FFN block**.

**What gets split in a transformer:**
- W_Q, W_K, W_V — split by attention heads (natural split, heads are independent)
- W_O (output projection) — split row-wise, results summed
- FFN W_1 (up-projection) — split column-wise
- FFN W_2 (down-projection) — split row-wise, results summed

**What's NOT split:** Embeddings (replicated), LayerNorm (tiny, replicated), biases (tiny).

**Communication:** All-reduce or all-gather within every layer. Requires fast NVLink (900 GB/s), not feasible over network.

**Pros:** Reduces per-GPU memory for individual layers. Enables very large d_model.

**Cons:** GPUs must communicate within each layer (not just at end of step). Needs NVLink.

**When to use:** Individual layers are too large for one GPU. Always within a single node.

---

### 3. Pipeline Parallelism

**What's split:** Model layers (each GPU holds a consecutive group of layers)

**How it works:**
1. Split model into stages (groups of consecutive layers): GPU 0 gets layers 0–19, GPU 1 gets layers 20–39, etc.
2. Split the batch into micro-batches
3. Micro-batch flows through the pipeline: GPU 0 → GPU 1 → GPU 2 → GPU 3
4. Once GPU 0 finishes micro-batch 1, it immediately starts micro-batch 2
5. All GPUs eventually work concurrently on different micro-batches

**Pipeline bubble:** GPUs are idle at startup (waiting for data to arrive) and at the end.

#### The Bubble Problem (Naive Schedule)

With 4 stages, all forwards happen first, then all backwards:

```
Time →   1    2    3    4    5    6    7    8
GPU 0: [F0] [  ] [  ] [  ] [F1] [  ] [  ] [  ]  ...
GPU 1: [  ] [F0] [  ] [  ] [  ] [F1] [  ] [  ]  ...
GPU 2: [  ] [  ] [F0] [  ] [  ] [  ] [F1] [  ]  ...
GPU 3: [  ] [  ] [  ] [F0] [  ] [  ] [  ] [F1]  ...
         ↑ idle ↑        ↑ idle ↑
```

GPU 3 sits idle for 3 time steps waiting for micro-batch 0 to arrive. GPU 0 sits idle waiting for backward passes to flow back up. These idle gaps = bubble.

#### Bubble Fraction Formula

```
Bubble fraction = (p - 1) / (p - 1 + m)
```

Where p = number of stages, m = number of micro-batches.

Examples:
- p=4, m=8:   bubble = 3/11 = 27% (bad)
- p=4, m=32:  bubble = 3/35 = 8.6%
- p=4, m=64:  bubble = 3/67 = 4.5% (acceptable)

More micro-batches → smaller bubble.

#### 1F1B Schedule (One Forward, One Backward)

Reduces the bubble by interleaving forward and backward passes:

**Phase 1 — Warmup:** Fill the pipeline with forward passes until the last GPU has work:
```
Time →   1    2    3    4
GPU 0: [F0] [F1] [F2] [F3]
GPU 1: [  ] [F0] [F1] [F2]
GPU 2: [  ] [  ] [F0] [F1]
GPU 3: [  ] [  ] [  ] [F0]
```

**Phase 2 — Steady state:** Each GPU alternates one forward + one backward. All GPUs stay busy:
```
GPU 0: [F4][B0] [F5][B1] [F6][B2] ...
GPU 1: [F3][B0] [F4][B1] [F5][B2] ...
GPU 2: [F2][B0] [F3][B1] [F4][B2] ...
GPU 3: [F1][B0] [F2][B1] [F3][B2] ...
```

**Phase 3 — Cooldown:** Drain remaining backward passes.

The bubble only exists during warmup and cooldown — the steady state has zero idle time. This is why 1F1B is standard in practice (used by Megatron-LM, DeepSpeed, etc.).

**Communication:** Only between adjacent GPUs (point-to-point). GPU 0 sends activations to GPU 1, GPU 1 to GPU 2, etc. Minimal volume.

**Pros:** Each GPU only holds a fraction of the layers. Low communication (adjacent only). Works across nodes.

**Cons:** Pipeline bubble (idle time). More complex scheduling.

**When to use:** Model has many layers that collectively don't fit on one GPU. Combined with tensor parallelism.

---

### 4. Sequence Parallelism

**What's split:** The token sequence dimension (each GPU handles a chunk of positions)

**Why it's needed:** Attention memory scales as O(seq²). For seq_len=32768 with 32 heads in FP16, attention scores alone = 64 GB. Exceeds a single A100.

#### Understanding the Memory Problem: Activations

"Activations" = all intermediate results stored during the forward pass (needed for backward to compute gradients). In one transformer block:

| Activation | Shape | Scaling |
|-----------|-------|---------|
| Input to layer | (batch, seq, d_model) | Linear with seq |
| Q, K, V matrices | (batch, heads, seq, d_head) | Linear with seq |
| **Attention scores** | **(batch, heads, seq, seq)** | **Quadratic with seq** |
| Attention output | (batch, seq, d_model) | Linear with seq |
| FFN intermediate | (batch, seq, d_ff) | Linear with seq |
| Residual connections | (batch, seq, d_model) | Linear with seq |

Most activations scale linearly with seq_len (they have one seq dimension). The attention scores are the outlier — they scale **quadratically** (seq × seq) because every token attends to every other token.

**Attention score memory formula:**
```
mem_bytes = batch_size × n_heads × seq_len × seq_len × 2 (bytes for FP16)
```

Examples (batch=1, 32 heads, FP16):
- seq=2048:  1 × 32 × 2048 × 2048 × 2 = **512 MB**
- seq=8192:  1 × 32 × 8192 × 8192 × 2 = **8 GB**
- seq=32768: 1 × 32 × 32768 × 32768 × 2 = **128 GB** (exceeds any single GPU)

This is just the attention scores — not weights, gradients, or other activations. This is why sequence parallelism specifically targets the attention computation.

**How it works (ring attention):**
1. Split the sequence across GPUs: GPU 0 gets tokens 0–8191, GPU 1 gets 8192–16383, etc.
2. Each GPU computes attention for its local tokens
3. K and V blocks are passed around in a ring pattern so each GPU eventually sees all keys/values
4. Memory per GPU: O(seq²/N) instead of O(seq²)

**Communication:** Ring-style all-to-all passing of K/V blocks between GPUs.

**Pros:** Enables very long context (32K, 128K+ tokens) that would otherwise OOM.

**Cons:** Complex attention communication patterns. Every GPU still needs to "see" all tokens eventually.

**When to use:** Long sequence lengths where attention memory exceeds GPU capacity.

---

### 5. Expert Parallelism

**What's split:** MoE experts (different experts live on different GPUs)

**How MoE works:**
1. A router network decides which experts each token should visit (e.g., top-2 out of 8)
2. Each token only goes through its selected experts (sparse activation)
3. A 47B MoE model with 8 experts uses similar compute as a 13B dense model per token

**How expert parallelism works:**
1. Distribute experts across GPUs: GPU 0 gets experts 0–1, GPU 1 gets experts 2–3, etc.
2. Router assigns tokens to experts
3. All-to-all communication: tokens are sent to whichever GPU holds their assigned expert
4. Each GPU processes the tokens routed to its experts
5. All-to-all: results are sent back

**Load balancing challenge:** Some experts may get more tokens than others → some GPUs are overloaded while others idle. Auxiliary loss functions encourage balanced routing.

**Communication:** All-to-all (every GPU may need to send tokens to every other GPU).

**Pros:** Massive model capacity without proportional compute cost.

**Cons:** Load imbalance, all-to-all communication overhead, routing complexity.

**When to use:** MoE architectures where you want more capacity without proportionally more compute.

---

### How They Combine (5D Parallelism)

For training something like LLaMA-3 70B on 256 GPUs:

```
┌─────────────────────────────────────────────────────┐
│  Data Parallelism: 8 replicas                       │
│  ┌───────────────────────────────────────────────┐  │
│  │  Pipeline Parallelism: 4 stages               │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  Tensor Parallelism: 8 GPUs per stage   │  │  │
│  │  │  (within one NVLink node)               │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
+ Sequence parallelism for long contexts
+ Expert parallelism for MoE models
```

**Why this nesting:**
- Tensor parallelism within a node (needs NVLink, communicates every layer)
- Pipeline parallelism across nodes (only adjacent communication, tolerates slower network)
- Data parallelism across groups of nodes (one all-reduce per step, tolerates slowest network)

**Total GPUs = TP × PP × DP**
Example: 8 (tensor) × 4 (pipeline) × 8 (data) = 256 GPUs
