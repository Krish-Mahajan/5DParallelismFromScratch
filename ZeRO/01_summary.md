# Notebook 01: ZeRO Memory Calculator

## Core Idea

ZeRO (Zero Redundancy Optimizer) eliminates redundant memory by sharding model states across GPUs instead of replicating them. Three stages progressively shard more components: Stage 1 (optimizer), Stage 2 (+ gradients), Stage 3 (+ weights).

## The Memory Problem

Mixed precision training with Adam uses **16 bytes per parameter**:
```
fp16 weights:     2 bytes/param
fp16 gradients:   2 bytes/param
Optimizer states: 12 bytes/param  (fp32 master + Adam m + Adam v)
```

For 7B model: 16 × 7B = 112 GB per GPU. Doesn't fit on A100 (80 GB).
Standard data parallelism replicates EVERYTHING — adding GPUs doesn't help memory.

## ZeRO Stages — What Gets Sharded

| Stage | Weights | Gradients | Optimizer | Formula per GPU | 7B on 8 GPUs |
|-------|---------|-----------|-----------|-----------------|--------------|
| Standard DP | Full | Full | Full | 16Ψ | 104 GB ✗ |
| ZeRO-1 | Full | Full | Sharded/N | 4Ψ + 12Ψ/N | ~36 GB ✓ |
| ZeRO-2 | Full | Sharded/N | Sharded/N | 2Ψ + 14Ψ/N | ~25 GB ✓ |
| ZeRO-3 | Sharded/N | Sharded/N | Sharded/N | 16Ψ/N | ~13 GB ✓ |

## The Memory Floor Concept

Each stage has a minimum memory regardless of GPU count:
```
ZeRO-1 floor: 4Ψ = weights + gradients (26 GB for 7B)   ← can't go lower
ZeRO-2 floor: 2Ψ = weights only (13 GB for 7B)          ← can't go lower
ZeRO-3 floor: 16Ψ/N → 0 as N→∞ (no floor!)             ← keeps shrinking
```

Only ZeRO-3 continues to scale with more GPUs. ZeRO-1/2 flatten out.

## Communication Cost — The Tradeoff

| Stage | Communication/step | Relative to DP | Memory savings |
|-------|-------------------|----------------|----------------|
| Standard DP | 4Ψ bytes | 1.0× | None |
| ZeRO-1 | 4Ψ bytes | 1.0× (same!) | ~3× less memory |
| ZeRO-2 | 4Ψ bytes | 1.0× (same!) | ~4× less memory |
| ZeRO-3 | 6Ψ bytes | 1.5× | N× less memory |

**ZeRO-1 and ZeRO-2 are free lunches** — same communication as standard DP but much less memory.

**ZeRO-3 costs 50% more communication** because of the extra AllGather operations:
```
1. Forward AllGather:        2Ψ bytes  (gather weights for forward pass)
2. Backward AllGather:       2Ψ bytes  (gather weights for backward pass)
3. Backward Reduce-Scatter:  2Ψ bytes  (distribute gradient shards)
Total: 6Ψ bytes
```

## ZeRO-3: How It Works Without OOM During AllGather

Weights are gathered **one layer at a time**, not all at once:
```
AllGather layer 0 weights → forward layer 0 → discard layer 0
AllGather layer 1 weights → forward layer 1 → discard layer 1
...
```

Peak memory = own shard + 1-2 layers temporarily (prefetch next layer while computing current).

## Why ZeRO-1 Is So Powerful

```
Standard DP:  104 GB  (optimizer = 78 GB is 75% of total!)
ZeRO-1 (8 GPUs): 36 GB  (optimizer = 78/8 = 9.75 GB)
```

Sharding JUST the optimizer (the easiest change, no extra communication) cuts total memory by ~3×. This is the most common first step in production.

## For 7B Model on 8 GPUs

```
                 Memory/GPU    Fits A100?    Communication
Standard DP:     104 GB          No           4Ψ = 28 GB
ZeRO-1:          36 GB          Yes          4Ψ = 28 GB (same!)
ZeRO-2:          25 GB          Yes          4Ψ = 28 GB (same!)
ZeRO-3:          13 GB          Yes          6Ψ = 42 GB (1.5×)
```

## Interview Explanation

"ZeRO eliminates memory redundancy in data parallelism by sharding model states across GPUs. Stage 1 shards optimizer states (biggest win — 3× reduction, zero extra communication). Stage 2 also shards gradients. Stage 3 shards everything including weights, reducing memory linearly with GPU count, at the cost of 1.5× more communication from AllGather operations during forward and backward passes. These AllGathers happen layer-by-layer to avoid OOM."

## Parameter Count Formula

For a standard transformer:
```
num_params = num_layers × 12 × hidden_dim² + vocab_size × hidden_dim

Per layer (12H²):
  Attention: 4H² (W_q, W_k, W_v, W_o)
  FFN:       8H² (up-projection + down-projection)
```

Examples:
```
GPT-2:    12 × 12 × 768²  + 50257 × 768  ≈ 124M
LLaMA-7B: 32 × 12 × 4096² + 32000 × 4096 ≈ 6.6B
LLaMA-70B: 80 × 12 × 8192² + 32000 × 8192 ≈ 65B
```

## Activation Memory Formula

```
activation_bytes = seq_len × batch_size × hidden_dim × num_layers × (10 + 24 × seq_len / hidden_dim)
```

The `10` covers linear-scaling terms (norms, projections, FFN). The `24 × seq_len / hidden_dim` is the attention scores (quadratic in seq_len).

## Complete Memory Planner

```
Total per GPU = model_states + activations

Model states (depends on ZeRO stage):
  Standard DP: 16Ψ
  ZeRO-1:      4Ψ + 12Ψ/N
  ZeRO-2:      2Ψ + 14Ψ/N
  ZeRO-3:      16Ψ/N

Activations (depends on batch/seq):
  = seq × batch × hidden × layers × (10 + 24×seq/hidden) bytes
```

## Redundancy Factor

```
Redundancy = (per_GPU_memory × N_GPUs) / unique_model_data

Standard DP on 8 GPUs: 8.0× (8 full copies of everything!)
ZeRO-1 on 8 GPUs:     2.8× (optimizer sharded, rest replicated)
ZeRO-2 on 8 GPUs:     2.0× (only weights replicated)
ZeRO-3 on 8 GPUs:     1.0× (zero redundancy!)
```

## What Comes Next

Implement ZeRO from scratch — manually shard optimizer states, gradients, and weights, and implement the AllGather/Reduce-Scatter communication patterns.
