# Notebook 01: Activation Checkpointing (Gradient Checkpointing)

## Core Idea

Trade compute for memory: discard intermediate activations during forward pass, recompute them during backward pass. Reduces activation memory from O(L) to O(√L) at the cost of ~33% more compute.

## Why Activations Dominate Memory

During forward, PyTorch saves every layer's output for backward (chain rule needs them). Memory grows linearly with depth:

```
hidden=2048, batch=32, seq=512:
  4 layers:  ~1500 MB activations
  12 layers: ~4600 MB
  24 layers: ~9200 MB  ← linear growth (~380 MB/layer)
```

Each layer stores ~3 tensors: input to linear, input to norm, input to activation function.

## How Checkpointing Works

**Normal:** Save all → use during backward
```
Forward:  layer1 → save a1 → layer2 → save a2 → ... → layerL → save aL
Backward: use aL → grad_L → use a(L-1) → grad_(L-1) → ... → use a1 → grad_1
Memory: O(L)
```

**Checkpointed:** Save only segment inputs → recompute rest during backward
```
Forward:  segment1(save input only) → segment2(save input only) → ...
Backward: recompute segment2 activations → grads → free → recompute segment1 → grads → free
Memory: O(√L) with √L segments
```

## From-Scratch Implementation

Custom autograd Function with two key methods:

**Forward:** Run layers with `torch.no_grad()` (don't save intermediates), only save segment input.

**Backward:** Re-run forward with `torch.enable_grad()` to reconstruct activations, then use `torch.autograd.grad()` to compute gradients.

```python
@staticmethod
def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    x = x.detach().requires_grad_(True)

    with torch.enable_grad():
        y = x
        for layer in ctx.layers:
            y = layer(y)

    grads = torch.autograd.grad(y, x, grad_output)
    return (grads[0],) + (None,) * len(ctx.layers)
```

**Critical:** Produces mathematically identical gradients — not an approximation.

## PyTorch's Built-in API

```python
from torch.utils.checkpoint import checkpoint

# In forward:
x = checkpoint(layer, x, use_reentrant=False)
```

One line. Handles edge cases (RNG state, nested checkpointing). Use `use_reentrant=False` (newer, recommended).

## Measured Results (H100, 12-layer transformer, hidden=512)

**seq_len=256:**
```
Standard:              1555 MB, 27 ms
Full Checkpointing:     827 MB, 44 ms  (47% memory saved, 59% overhead)
```

**seq_len=1024:**
```
Standard:              7496 MB, 78 ms
Full Checkpointing:    1862 MB, 100 ms  (75% memory saved, 27% overhead)
Selective:             3622 MB, 92 ms   (52% memory saved, 17% overhead)
```

Larger models/sequences → overhead converges toward theoretical 33%, savings increase.

## Selective Checkpointing

Only checkpoint the memory-hungry parts (attention), let cheap parts (FFN) run normally:

```python
def forward(self, x):
    for block in self.blocks:
        x = checkpoint(self._attn_forward, block, x, use_reentrant=False)  # ← checkpointed
        x = x + block['ff'](block['ln2'](x))                               # ← normal
    return x
```

**Why it helps:** Attention stores (batch, heads, seq, seq) — quadratic in seq_len. FFN stores (batch, seq, 4H) — much smaller at long sequences.

```
seq=256:  attention ≈ FFN size → selective saves less
seq=1024: attention >> FFN     → selective saves nearly as much as full, with half the overhead
seq=2048: attention >>> FFN    → selective ≈ full savings, much less overhead
```

## When to Use Which

| Strategy | Memory Savings | Compute Overhead | When to Use |
|----------|---------------|-----------------|-------------|
| No checkpointing | 0% | 0% | Memory isn't the bottleneck |
| Selective | 30-50% | 15-20% | Have some headroom, want speed |
| Full checkpointing | 50-75% | 25-60% | Close to OOM, need every MB |

## Key Takeaways

- Activations are #1 memory consumer during training (can exceed weights + optimizer for large batches)
- Checkpointing = identical gradients, just slower (not an approximation)
- Compute overhead decreases (relative) as model size increases
- Selective checkpointing is the practical default in production (LLaMA, GPT use it)
- `torch.utils.checkpoint` handles all edge cases — use it instead of manual implementation

## What Comes Next

Gradient accumulation — simulate large batch sizes without holding the full batch in memory.
