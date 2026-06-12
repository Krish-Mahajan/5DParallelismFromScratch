# Notebook 03: Data Parallelism

## Core Idea

Replicate the full model on every GPU. Split the training data across GPUs. Each GPU computes gradients on its shard, then all GPUs average their gradients (AllReduce) and apply the same update. Models stay in sync.

## The 5-Step Workflow

```
1. REPLICATE:  Copy model to all N GPUs (identical weights)
2. SHARD:      Split batch into N chunks (each GPU gets batch/N samples)
3. COMPUTE:    Each GPU runs forward + backward independently on its shard
4. ALLREDUCE:  Average gradients across all GPUs
5. UPDATE:     Every GPU applies the same averaged gradient → weights stay in sync
```

**Key invariant:** Same averaged gradient + same optimizer state = identical weights on all GPUs after every step.

## Why It Works Mathematically

The mean of local gradients equals the gradient of the full batch:

```
GPU 0 gradient: ∇L(samples 0-7)
GPU 1 gradient: ∇L(samples 8-15)
GPU 2 gradient: ∇L(samples 16-23)
GPU 3 gradient: ∇L(samples 24-31)

Average = (∇L(0-7) + ∇L(8-15) + ∇L(16-23) + ∇L(24-31)) / 4
        = ∇L(samples 0-31)  ← identical to full batch gradient
```

## Ring AllReduce — How GPUs Communicate

**The problem:** Every GPU needs the average of all gradients. Naive approach (send everything to GPU 0) creates a bottleneck.

**The solution — Ring AllReduce:** Pass chunks around a circle, each GPU adds its piece as data passes through.

**Example (4 GPUs, gradient split into 4 chunks):**

**Phase 1 — Reduce-Scatter (N-1 rounds):**
Each GPU sends a chunk to its neighbor, neighbor adds it to its own.

```
Start:
  GPU 0: [A0, B0, C0, D0]    GPU 1: [A1, B1, C1, D1]
  GPU 2: [A2, B2, C2, D2]    GPU 3: [A3, B3, C3, D3]

After 3 rounds: each GPU holds the COMPLETE SUM of one chunk
  GPU 0: [_, B_sum, _, _]    GPU 1: [_, _, C_sum, _]
  GPU 2: [_, _, _, D_sum]    GPU 3: [A_sum, _, _, _]
```

**Phase 2 — All-Gather (N-1 more rounds):**
Pass the complete chunks around (just copy, no adding).

```
After 3 more rounds: every GPU has ALL summed chunks
  GPU 0: [A_sum, B_sum, C_sum, D_sum] ✓
  GPU 1: [A_sum, B_sum, C_sum, D_sum] ✓
  GPU 2: [A_sum, B_sum, C_sum, D_sum] ✓
  GPU 3: [A_sum, B_sum, C_sum, D_sum] ✓
```

**Why ring is efficient:**
- Every GPU sends AND receives simultaneously (no idle time, no bottleneck)
- Total data per GPU: 2 × (N-1)/N × gradient_size — approaches 2× data regardless of N
- All GPUs fully utilized at every round

## Communication Overhead

```
Total data moved per GPU = 2 × (N-1)/N × gradient_size (FP16)

For 1.3B model (grad = 2.6 GB in FP16) at 25 GB/s (NVLink):
  2 GPUs:  ~2.6 GB moved, ~4 ms
  8 GPUs:  ~4.6 GB moved, ~7 ms
  32 GPUs: ~4.8 GB moved, ~7.5 ms  (barely increases with N!)
```

## Scaling Efficiency

```
Efficiency = compute_time / (compute_time + communication_time)

1 GPU:  100% (no communication)
2 GPUs: ~93%
8 GPUs: ~88%
32 GPUs: ~87%
```

**Key insight:** Larger models scale better. Compute grows as O(params²) but communication grows as O(params). A 70B model on 64 GPUs has better efficiency than a 1B model on 64 GPUs.

## PyTorch DDP Pattern (Production Code)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Training loop is normal — DDP handles AllReduce automatically
for batch in dataloader:
    loss = model(batch)
    loss.backward()      # ← DDP hooks trigger AllReduce here
    optimizer.step()
    optimizer.zero_grad()
```

DDP overlaps communication with backward computation — starts AllReducing early layers while later layers are still computing gradients.

## How All Three Techniques Compose

```
Single GPU, memory-limited:
  1. Activation checkpointing → reduce activation memory (trade compute)
  2. Gradient accumulation → simulate larger batch without memory

Multiple GPUs:
  3. Data parallelism → true parallel speedup, same effective batch across GPUs

Combined (production):
  Each GPU uses checkpointing + accumulation internally,
  then AllReduces after K accumulation steps.
  Effective batch = micro_bs × K × N_GPUs
```

## Pros and Cons

| Pros | Cons |
|------|------|
| Simple to implement (DDP) | Every GPU holds full model (memory not saved) |
| Near-linear speedup | Communication grows with model size |
| Works with any model that fits on 1 GPU | Requires identical model copies |
| Mathematically identical to single-GPU | AllReduce bandwidth can bottleneck |

## When to Use

- Model fits on one GPU, but training is too slow
- Want to scale batch size without gradient accumulation's K× slowdown
- First parallelism strategy to try (simplest, most robust)

## When NOT Sufficient

- Model doesn't fit on one GPU → need tensor/pipeline parallelism or ZeRO
- Communication bandwidth is limited (cross-node) → combine with gradient accumulation to reduce AllReduce frequency

## What Comes Next

ZeRO optimizer — shard optimizer states/gradients/weights across GPUs to reduce per-GPU memory (unlike data parallelism which replicates everything).
