# Notebook 02: ZeRO from Scratch

## Core Idea

Implement ZeRO Stages 1 and 2 manually by simulating multiple GPUs on one device. Understand exactly how optimizer sharding and gradient sharding work, and verify they produce identical results to standard data parallelism.

## The Three Communication Primitives

### AllReduce — everyone gets the full average

```
Before:
  Rank 0: [1, 2, 3, 4]    Rank 1: [2, 4, 6, 8]    Rank 2: [3, 6, 9, 12]    Rank 3: [4, 8, 12, 16]

Step 1 — Sum element-wise:  [10, 20, 30, 40]
Step 2 — Divide by N=4:     [2.5, 5.0, 7.5, 10.0]
Step 3 — Copy to all ranks:

After:
  Rank 0: [2.5, 5.0, 7.5, 10.0]   ← full result
  Rank 1: [2.5, 5.0, 7.5, 10.0]   ← same full result
  Rank 2: [2.5, 5.0, 7.5, 10.0]
  Rank 3: [2.5, 5.0, 7.5, 10.0]

Total stored: 16 elements (4× redundancy)
Used by: Standard data parallelism
```

### ReduceScatter — average, then each rank keeps only its chunk

```
Before: same 4 tensors

Step 1 — Same sum/average: [2.5, 5.0, 7.5, 10.0]
Step 2 — Split and distribute: each rank gets 1/N of the result

After:
  Rank 0: [2.5]        ← only its portion
  Rank 1: [5.0]
  Rank 2: [7.5]
  Rank 3: [10.0]

Total stored: 4 elements (zero redundancy!)
Used by: ZeRO-2 for gradients
```

### AllGather — each rank contributes its chunk, all get the full tensor

```
Before:
  Rank 0: [2.5]    Rank 1: [5.0]    Rank 2: [7.5]    Rank 3: [10.0]

Step 1 — Each rank broadcasts its chunk to all others
Step 2 — Each rank concatenates all chunks in order

After:
  Rank 0: [2.5, 5.0, 7.5, 10.0]   ← full reconstructed tensor
  Rank 1: [2.5, 5.0, 7.5, 10.0]
  Rank 2: [2.5, 5.0, 7.5, 10.0]
  Rank 3: [2.5, 5.0, 7.5, 10.0]

Used by: ZeRO-2 after optimizer step (reconstruct full weights)
         ZeRO-3 before forward/backward (gather sharded weights)
```

### The Key Relationship

```
ReduceScatter + AllGather = AllReduce
```

ZeRO's trick: split AllReduce into two halves, insert the optimizer step in between — when each rank only needs its portion.

```
Standard DP:  AllReduce → optimizer updates ALL params (redundant × N)
ZeRO-2:       ReduceScatter → optimizer updates only 1/N params → AllGather
```

## Helper Functions: Flatten/Unflatten

ZeRO treats ALL parameters as one big flat vector — easier to split into equal chunks for sharding.

```python
flatten_params(model)  → [W1, b1, W2, b2, W3, b3] flattened to one 1D tensor (6792 elements)
flatten_grads(model)   → same for gradients
unflatten_to_model(flat, model) → reshapes flat vector back into model's parameter tensors
```

Why: ReduceScatter splits this flat vector into N equal chunks. Rank 0 gets elements 0-1697, Rank 1 gets 1698-3395, etc. After AllGather reconstructs the full vector, unflatten copies it back into the model's weight matrices.

## Standard Data Parallelism (Baseline)

```python
for step in range(num_steps):
    # Each rank: forward + backward on its data shard (different data, same model)
    for rank in range(N):
        grads[rank] = compute_gradient(model[rank], data[rank])

    # AllReduce: every rank gets the FULL averaged gradient
    all_grads = [flatten_grads(m) for m in models]
    avg_grads = comm.allreduce(all_grads)

    # Write averaged gradient back into model, step optimizer
    for rank in range(N):
        unflatten_grad(avg_grads[rank], models[rank])
        optimizers[rank].step()  # every rank updates ALL params (redundant!)
```

Memory per rank: full weights + full gradients + full optimizer = 16Ψ

The waste: every rank stores full model (×N redundancy), does full optimizer step (×N redundant work).

## ZeRO Stage 1: Partitioned Optimizer

### ZeROStage1Optimizer — what each rank stores

```python
class ZeROStage1Optimizer:
    def __init__(self, flat_params, world_size, rank, ...):
        self.chunk_size = total_params // world_size
        start = rank * self.chunk_size
        end = start + self.chunk_size

        # ONLY stores states for its 1/N partition
        self.master_weights = flat_params[start:end].float().clone()
        self.m = torch.zeros_like(self.master_weights)
        self.v = torch.zeros_like(self.master_weights)
```

```
1000 params, 4 ranks:
  Rank 0: master/m/v for params[0:250]     ← 250 × 12 bytes
  Rank 1: master/m/v for params[250:500]
  Rank 2: master/m/v for params[500:750]
  Rank 3: master/m/v for params[750:1000]

Standard Adam: 1000 × 12 = 12,000 bytes/rank
ZeRO-1 Adam:  250 × 12 = 3,000 bytes/rank  (4× less)
```

### Training loop step by step

```python
# 1. Forward + backward (same as standard DP)
for rank in range(world_size):
    loss = model[rank](data[rank]).backward()

# 2. AllReduce gradients (same as standard DP — every rank gets full gradient)
all_grads = [flatten_grads(m) for m in models]
avg_grads = comm.allreduce(all_grads)

# 3. Each rank updates ONLY its partition
for rank in range(world_size):
    grad_partition = avg_grads[rank][start:end]  # slice out my 1/N
    updated = optimizers[rank].step(grad_partition)  # Adam on just that slice
    updated_partitions.append(updated)

# 4. AllGather — share updated shards, reconstruct full model
full_weights_list = comm.allgather(updated_partitions)

# 5. Write full weights back into all models
for rank in range(world_size):
    unflatten_to_model(full_weights_list[rank], models[rank])
```

**Step 3 is the key difference:** instead of every rank running Adam on all 1000 params (redundant), each rank runs Adam on only 250 params. The other 750 params' gradients are thrown away — other ranks handle those.

Memory per rank: full weights + full gradients + 1/N optimizer = 4Ψ + 12Ψ/N

## ZeRO Stage 2: Partitioned Gradients + Optimizer

```python
for step in range(num_steps):
    # Same forward + backward
    for rank in range(N):
        grads[rank] = compute_gradient(model[rank], data[rank])

    # KEY DIFFERENCE: ReduceScatter instead of AllReduce
    # Each rank gets ONLY its gradient shard (not the full gradient)
    grad_shards = comm.reduce_scatter(grads)  # rank i gets grad for its params only

    # Each rank updates only ITS shard (has the gradient it needs)
    for rank in range(N):
        optimizer_shard[rank].step(grad_shards[rank])

    # AllGather: reconstruct full weights
    updated_shards = [get_params_shard(rank) for rank in range(N)]
    full_params = comm.allgather(updated_shards)
```

Memory per rank: full weights + 1/N gradients + 1/N optimizer = 2Ψ + 14Ψ/N

## Why All Approaches Produce Identical Results

All three compute the same mathematical operations:
1. Same forward pass (same weights)
2. Same backward pass (same data sharding)
3. Same gradient averaging (AllReduce = ReduceScatter + AllGather)
4. Same optimizer update (same averaged gradient applied to same params)

The only difference is **where** data is stored — not what is computed. Loss curves should be identical.

## Memory Comparison (verified empirically)

```
Standard DP:  16Ψ per rank (full replication)
ZeRO-1:      4Ψ + 12Ψ/N (optimizer sharded, rest replicated)
ZeRO-2:      2Ψ + 14Ψ/N (optimizer + gradients sharded)
```

## Communication Volume (Measured Empirically)

```
Standard DP:  2P × bytes  (AllReduce only)              = 54,336 bytes
ZeRO-1:      3P × bytes  (AllReduce + AllGather)        = 81,504 bytes (1.5× DP!)
ZeRO-2:      2P × bytes  (ReduceScatter + AllGather)    = 54,336 bytes (same as DP!)
```

**Corrected understanding:**
- ZeRO-2 is truly free — ReduceScatter + AllGather = AllReduce (same total bytes, just split into two steps)
- ZeRO-1 actually costs 1.5× more (extra AllGather for weight shards after optimizer step)
- In practice, ZeRO-1's AllGather can be overlapped with next step's forward → effective cost is similar

| Approach | Operations | Relative Cost |
|----------|-----------|---------------|
| Standard DP | AllReduce(grads) | 1.0× |
| ZeRO-1 | AllReduce(grads) + AllGather(weights) | 1.5× |
| ZeRO-2 | ReduceScatter(grads) + AllGather(weights) | 1.0× (same as DP!) |
| ZeRO-3 | 2×AllGather(weights) + ReduceScatter(grads) | 1.5× |

## ZeRO-1 vs ZeRO-2: The One-Line Difference

```python
# ZeRO-1:
avg_grads = comm.allreduce(all_grads)          # every rank gets FULL gradient
grad_partition = avg_grads[rank][start:end]     # then slices out its portion (wastes memory!)

# ZeRO-2:
grad_partitions = comm.reduce_scatter(all_grads)  # each rank gets ONLY its portion directly
```

ZeRO-1 holds the full gradient temporarily then discards 75%. ZeRO-2 never holds the other 75% in the first place.

## Does ZeRO-1 Alone Make Sense in Practice?

**Yes — ZeRO-1 is the most common first step.** Here's why:

**Use ZeRO-1 when:**
- Simplest change: just wrap the optimizer, no training loop changes
- Full gradient needed for: gradient clipping (global norm), monitoring, some overlap schemes
- Memory isn't critically tight (36 GB fits on 80 GB GPU with room for activations)
- Framework has better AllReduce than ReduceScatter performance

**Upgrade to ZeRO-2 when:**
- Need those extra GBs (for larger batch or more activations)
- Already using DeepSpeed (one config flag change)
- Very large model where gradient memory (2Ψ = 13 GB for 7B) matters

**The practical progression:**
```
Step 1: ZeRO-1 (wrap optimizer, biggest bang, easiest)
Step 2: ZeRO-2 (if still OOM, one config change)
Step 3: ZeRO-3 (if model itself doesn't fit, 1.5× communication cost)
```

## What Comes Next

Use DeepSpeed's production ZeRO implementation — configure stages, offloading, and train real models.
