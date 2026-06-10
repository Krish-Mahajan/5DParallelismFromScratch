# Tensor Parallelism Deep Dive

## Core Idea

Split individual weight matrices across GPUs. Each GPU holds a slice of the layer and computes part of the result. Used when a single layer's weights are too large for one GPU.

## Two Approaches: Column-Parallel vs Row-Parallel

### Column-Parallel (split output dimension)

Each GPU computes a **subset of output features completely**, then concatenate.

**Example:** W (4×4), x (1,4), 2 GPUs

```
W split by rows (output features):
  GPU 0: W[0:2, :] = [[a,b,c,d], [e,f,g,h]]     ← output features 0,1
  GPU 1: W[2:4, :] = [[i,j,k,l], [m,n,o,p]]     ← output features 2,3

Input: FULL x on both GPUs = [1, 2, 3, 4]

GPU 0 computes:
  [1,2,3,4] @ [[a,b,c,d], [e,f,g,h]]^T = [1a+2b+3c+4d, 1e+2f+3g+4h]
  → (1, 2): features 0,1 COMPLETE

GPU 1 computes:
  [1,2,3,4] @ [[i,j,k,l], [m,n,o,p]]^T = [1i+2j+3k+4l, 1m+2n+3o+4p]
  → (1, 2): features 2,3 COMPLETE

Combine: CONCATENATE → [f0, f1, f2, f3] = full output ✓
```

### Row-Parallel (split input dimension)

Each GPU computes a **partial sum of ALL output features**, then sum.

**Example:** W (4×4), x (1,4), 2 GPUs

```
W split by columns (input features):
  GPU 0: W[:, 0:2] = [[a,b], [e,f], [i,j], [m,n]]    ← first 2 input connections
  GPU 1: W[:, 2:4] = [[c,d], [g,h], [k,l], [o,p]]    ← last 2 input connections

Input: SPLIT x
  GPU 0: x[0:2] = [1, 2]
  GPU 1: x[2:4] = [3, 4]

GPU 0 computes:
  [1,2] @ [[a,b], [e,f], [i,j], [m,n]]^T = [1a+2b, 1e+2f, 1i+2j, 1m+2n]
  → (1, 4): PARTIAL sum (only first 2 inputs)

GPU 1 computes:
  [3,4] @ [[c,d], [g,h], [k,l], [o,p]]^T = [3c+4d, 3g+4h, 3k+4l, 3o+4p]
  → (1, 4): PARTIAL sum (only last 2 inputs)

Combine: SUM → [1a+2b+3c+4d, 1e+2f+3g+4h, ...] = full output ✓
```

Why summing works: a dot product of length 4 = sum of two dot products of length 2.

### Side-by-Side Comparison

| | Column-Parallel | Row-Parallel |
|---|---|---|
| W split | Rows (output dim) | Columns (input dim) |
| x on each GPU | Full (same everywhere) | Split (each GPU gets a slice) |
| Each GPU output | Subset of features (complete) | All features (partial sum) |
| Output shape per GPU | (batch, d_out/N) | (batch, d_out) |
| Combine operation | **Concatenate** (all-gather) | **Sum** (all-reduce) |

## Communication Primitives

### All-Gather

Every GPU has a **piece**. After: every GPU has **all pieces concatenated**.

```
Before:                    After all-gather:
  GPU 0: [A]                GPU 0: [A, B, C, D]
  GPU 1: [B]                GPU 1: [A, B, C, D]
  GPU 2: [C]                GPU 2: [A, B, C, D]
  GPU 3: [D]                GPU 3: [A, B, C, D]
```

Used after column-parallel: concatenate subset outputs.

### All-Reduce

Every GPU has a **full-sized partial value**. After: every GPU has the **sum**.

```
Before:                    After all-reduce (sum):
  GPU 0: [1, 2, 3, 4]       GPU 0: [8, 12, 16, 16]
  GPU 1: [5, 6, 7, 8]       GPU 1: [8, 12, 16, 16]
  GPU 2: [2, 1, 4, 3]       GPU 2: [8, 12, 16, 16]
  GPU 3: [0, 3, 2, 1]       GPU 3: [8, 12, 16, 16]
```

Used after row-parallel: sum partial outputs. Also used in data parallelism: average gradients.

## The Megatron-LM Pattern: Why Both Are Used Together

### The Insight: Column-parallel output is Row-parallel's input

```
FFN Layer 1 (column-parallel):
  Input x: full, same on all GPUs
  Output: each GPU has (batch, seq, d_ff/N)  ← split across feature dim

         ↓ NO COMMUNICATION — output is already split correctly! ↓

FFN Layer 2 (row-parallel):
  Input: split features from Layer 1 (already on the right GPU)
  Output: partial sums → all-reduce → full output on every GPU
```

Pairing them = only **one all-reduce** for two layers combined. The all-gather between them is eliminated entirely.

### Applied to a Full Transformer Block

```
Attention block:
  W_Q, W_K, W_V (column-parallel) → split by heads (natural, heads are independent)
                                     ↓ no communication
  W_O (row-parallel)               → all-reduce
                                     = 1 all-reduce

FFN block:
  FFN Layer 1 (column-parallel)    → split expanded features
                                     ↓ no communication
  FFN Layer 2 (row-parallel)       → all-reduce
                                     = 1 all-reduce

Total per transformer block: 2 all-reduces
```

### Why This Nesting is Efficient

Without the pairing trick:
```
Column-parallel W_Q: → all-gather (cost)
Column-parallel W_K: → all-gather (cost)
Column-parallel W_V: → all-gather (cost)
Column-parallel W_O: → all-gather (cost)
Column-parallel FFN1: → all-gather (cost)
Column-parallel FFN2: → all-gather (cost)
= 6 communications per block
```

With Megatron pairing:
```
Column → Row (attention): 1 all-reduce
Column → Row (FFN):       1 all-reduce
= 2 communications per block (3× fewer!)
```

## What Gets Split in Practice

| Component | Split Strategy | Why |
|-----------|---------------|-----|
| W_Q, W_K, W_V | Column-parallel (by heads) | Heads are independent, natural split |
| W_O | Row-parallel | Takes split head outputs, sums |
| FFN Layer 1 (up) | Column-parallel | Split expanded features |
| FFN Layer 2 (down) | Row-parallel | Takes split features, sums |
| Embeddings | Replicated | Tiny, not worth splitting |
| LayerNorm | Replicated | Tiny, not worth splitting |

## Requirements and Constraints

- **Needs NVLink** (900 GB/s) — communication happens WITHIN every layer, not just at step boundaries. Network bandwidth (25-100 GB/s) is too slow.
- **Always within a single node** (8 GPUs) — never across machines.
- **d_model must be divisible by N** (number of GPUs).

## Interview Explanation

"Tensor parallelism splits individual weight matrices across GPUs. There are two approaches: column-parallel splits the output dimension (each GPU computes some features completely, then concatenate), and row-parallel splits the input dimension (each GPU computes a partial sum of all features, then sum). In practice, Megatron-LM pairs them — column-parallel for the first layer, row-parallel for the second — which eliminates the communication between them. A full transformer block needs only 2 all-reduces total. It requires NVLink bandwidth so it's always used within a single node."
