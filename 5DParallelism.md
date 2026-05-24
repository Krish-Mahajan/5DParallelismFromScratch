# 5D Training Parallelism

## Assembly Line Analogy

Think of training a large model like building cars on an assembly line:
- **Pipeline parallelism** = the assembly line stations (frame → engine → paint → interior)
- **Tensor parallelism** = splitting one station's job among multiple workers (4 painters, each painting one side of the same car)
- **Data parallelism** = running multiple identical assembly lines, each building a different car, then sharing what they learned

---

## 1. Data Parallelism

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

## 2. Tensor Parallelism

**What's split:** Individual weight matrices (each GPU holds a slice of the layer)

**How it works (column-parallel):**
1. Weight matrix W of shape (d_out, d_in) is split into row chunks across N GPUs
2. GPU i gets W[i*chunk:(i+1)*chunk, :] — a slice of the output dimension
3. Each GPU computes its partial output: y_i = x @ W_i^T → shape (batch, d_out/N)
4. All-gather: concatenate partial outputs → full (batch, d_out) result

**How it works (row-parallel):**
1. Weight matrix W is split along columns (input dimension)
2. Input x is also split correspondingly
3. Each GPU computes: y_i = x_i @ W_i^T → shape (batch, d_out) but partial
4. All-reduce (sum): y = y_0 + y_1 + ... + y_{N-1}

**Megatron-LM pattern:** Column-parallel (QKV, FFN up) followed by row-parallel (output proj, FFN down). This requires only one all-reduce per attention block and one per FFN block.

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

## 3. Pipeline Parallelism

**What's split:** Model layers (each GPU holds a consecutive group of layers)

**How it works:**
1. Split model into stages (groups of consecutive layers): GPU 0 gets layers 0–19, GPU 1 gets layers 20–39, etc.
2. Split the batch into micro-batches
3. Micro-batch flows through the pipeline: GPU 0 → GPU 1 → GPU 2 → GPU 3
4. Once GPU 0 finishes micro-batch 1, it immediately starts micro-batch 2
5. All GPUs eventually work concurrently on different micro-batches

**Pipeline bubble:** GPUs are idle at startup (waiting for data to arrive) and at the end.

### The Bubble Problem (Naive Schedule)

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

### Bubble Fraction Formula

```
Bubble fraction = (p - 1) / (p - 1 + m)
```

Where p = number of stages, m = number of micro-batches.

Examples:
- p=4, m=8:   bubble = 3/11 = 27% (bad)
- p=4, m=32:  bubble = 3/35 = 8.6%
- p=4, m=64:  bubble = 3/67 = 4.5% (acceptable)

More micro-batches → smaller bubble.

### 1F1B Schedule (One Forward, One Backward)

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

## 4. Sequence Parallelism

**What's split:** The token sequence dimension (each GPU handles a chunk of positions)

**Why it's needed:** Attention memory scales as O(seq²). For seq_len=32768 with 32 heads in FP16, attention scores alone = 64 GB. Exceeds a single A100.

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

## 5. Expert Parallelism

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

## How They Combine (5D Parallelism)

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
