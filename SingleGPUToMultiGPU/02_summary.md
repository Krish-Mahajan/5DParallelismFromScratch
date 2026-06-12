# Notebook 02: Gradient Accumulation

## Core Idea

Simulate a large batch by running multiple small forward-backward passes and accumulating gradients before one optimizer step. Mathematically identical to a large batch, but uses only the memory of one micro-batch.

## The Problem

Memory scales linearly with batch size (activations grow):
```
Batch  4:   ~800 MB
Batch  8:  ~1200 MB
Batch 16:  ~2000 MB
Batch 32:  ~3600 MB
Batch 64:  ~6800 MB or OOM
```

You want batch=64 for stable gradients, but GPU only fits batch=16.

## The Solution

Run 4 micro-batches of 16, accumulate gradients, step once:

```python
optimizer.zero_grad()

for k in range(K):  # K = accumulation steps
    out = model(micro_batch[k])
    loss = F.cross_entropy(out, labels[k]) / K   # ← scale by 1/K!
    loss.backward()                               # ← gradients accumulate

optimizer.step()  # one update with accumulated gradient
```

Effective batch size = micro_batch_size × K = 16 × 4 = 64.

## Why `/K` Is Critical

`loss.backward()` ADDS to `.grad` (doesn't overwrite). Without `/K`, after K accumulations the gradient is K× too large:

```
Full batch gradient norm:      0.15
Accumulated (correct, /K):     0.15  ← matches ✓
Accumulated (WRONG, no /K):    0.60  ← 4× too large!
```

## The Complete Pattern

```
1. optimizer.zero_grad()          ← start of cycle
2. for k in range(K):
     loss = compute_loss(mini_batch_k)
     (loss / K).backward()        ← scale and accumulate
3. optimizer.step()               ← one update
4. (back to step 1)
```

**Never** call `zero_grad()` inside the loop — it erases accumulated gradients.

## Linear Scaling Rule

When effective batch size increases by K, scale learning rate by K:

```
Baseline:    BS=8,  LR=1e-3  → normal convergence
Large batch: BS=32, LR=1e-3  → converges slower (gradient too smooth relative to LR)
Scaled:      BS=32, LR=4e-3  → matches baseline convergence ✓
```

This is the "linear scaling rule" (Goyal et al., 2017). Larger batch = smoother gradient → need proportionally larger steps to compensate.

## Memory Savings

```
Without accumulation (BS=64): activations for 64 samples in memory simultaneously
With accumulation (BS=16, K=4): activations for only 16 samples at a time

Memory: same as BS=16
Gradient quality: same as BS=64
```

Only activation memory is reduced. Weights, gradients (accumulated), and optimizer states stay the same size.

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| No loss scaling | Gradient is K× too large | Divide loss by K |
| zero_grad inside loop | Erases accumulated gradients | Only zero_grad before the loop |
| BatchNorm + tiny micro-batch | Noisy batch statistics | Use LayerNorm (transformers already do) |
| Forgetting LR scaling | Slower convergence with larger effective batch | Scale LR by K |

## When NOT to Use Gradient Accumulation

| Situation | Why not? |
|-----------|----------|
| Model itself doesn't fit (weights+optimizer > GPU) | Batch=1 still OOMs — need ZeRO |
| Using BatchNorm with micro-batch < 8 | Noisy statistics corrupt training |
| Need maximum wall-clock speed | K× slower per optimizer step |
| Can afford more GPUs | Data parallelism gives true parallelism (faster) |
| Activations aren't the bottleneck | Nothing to save |

## Gradient Accumulation vs Data Parallelism

| | Gradient Accumulation | Data Parallelism |
|---|---|---|
| Hardware | 1 GPU | N GPUs |
| Speed | K× slower per update | ~1× (parallel) |
| Memory per GPU | Same as micro-batch | Same as micro-batch |
| Communication | None | All-reduce each step |
| Effective batch | micro_bs × K | micro_bs × N |

In practice, they're often combined: each GPU accumulates K steps, then all-reduce → effective batch = micro_bs × K × N_GPUs.

## What Comes Next

Data parallelism — replicate the model across multiple GPUs, split the data, all-reduce gradients. True parallel speedup instead of sequential accumulation.
