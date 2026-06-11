# Notebook 03: Will It Fit? — GPU Memory Calculator

## Core Idea

Before launching a training job, predict whether it will OOM. The calculator combines all four memory components (weights, gradients, optimizer, activations) and tells you the maximum batch size for your GPU.

## The Complete Formula

$$M_{\text{total}} = \underbrace{16P}_{\text{fixed (weights+grads+optimizer)}} + \underbrace{L \cdot B \cdot S \cdot H \cdot 34}_{\text{activations (variable)}}$$

Only activations depend on batch size. Everything else is fixed for a given model.

## Maximum Batch Size Formula

$$B_{\max} = \frac{M_{\text{GPU}} \times 0.88 - 16P}{L \cdot S \cdot H \cdot 34}$$

The 0.88 accounts for ~12% PyTorch/CUDA overhead (context, fragmentation, temp buffers, caching allocator).

If B_max ≤ 0: the model's fixed cost alone exceeds GPU memory → need multi-GPU or ZeRO.

## The Calculator Function

```python
result = will_it_fit(
    num_params=7e9,
    num_layers=32,
    hidden_dim=4096,
    num_heads=32,
    batch_size=1,
    seq_len=2048,
    gpu_memory_gb=80,
    precision="mixed",
    activation_checkpointing=False,
)
# Returns: weights_gb, gradients_gb, optimizer_gb, activations_gb, total_gb, fits, max_batch_size
```

## Key Results: Compatibility Matrix

```
Model              T4 (16GB)   RTX 3090    A100-40    A100-80
GPT-2 (124M)      B<=32+      B<=32+      B<=32+     B<=32+
GPT-2 XL (1.5B)   -           B<=1        B<=4       B<=8
LLaMA-7B           -           -           -          B=0 (fixed cost > 80GB!)
LLaMA-13B          -           -           -          -
LLaMA-70B          -           -           -          -
```

LLaMA-7B doesn't fit on any single GPU for training — the optimizer alone is 78 GB.

## Activation Checkpointing Impact

Reduces activation memory by ~√L factor (recomputes instead of storing):

```
LLaMA-7B, A100 80GB, S=2048:
  Without checkpointing: 28.5 GB activations, total 133 GB → doesn't fit
  With checkpointing:    5.0 GB activations, total 109 GB → still doesn't fit!

Problem: fixed model cost (104 GB) exceeds GPU regardless of activations.
Checkpointing only helps when activations are the bottleneck, not optimizer.
```

## When It Doesn't Fit — Decision Framework

```
1. Max batch size > 0 but too small?
   → Reduce batch size + gradient accumulation
   → Activation checkpointing (saves act memory, costs ~33% more compute)

2. Max batch size = 0 (model itself doesn't fit)?
   → ZeRO-1: shard optimizer across GPUs (biggest win, targets 78 GB for 7B)
   → ZeRO-2: also shard gradients
   → ZeRO-3: also shard weights
   → Tensor parallelism: split layers across GPUs (needs NVLink)
   → Pipeline parallelism: split layer groups across GPUs
```

## Inference vs Training Memory

```
Training:  Weights + Gradients + Optimizer + Activations = 16P + L*B*S*H*34
Inference: Weights + KV Cache only                       = 2P + KV cache

7B model:
  Training:  112+ GB  (needs 2+ A100s)
  Inference: ~15 GB   (fits on RTX 3090!)
```

Inference doesn't need gradients (no backward), optimizer (no updates), or full activations. Only weights + KV cache for generated sequence.

## The 12% Overhead — What It Covers

- CUDA context initialization (~200-500 MB)
- PyTorch caching allocator (holds freed blocks in pool)
- Memory fragmentation (unusable gaps between allocations)
- Temporary buffers (softmax denominators, etc.)
- NCCL communication buffers (if multi-GPU)
- Kernel workspace memory

## AdaFactor vs Adam

```
Adam:      12 bytes/param (master + m + v, all full matrices)
AdaFactor: ~8 bytes/param (factorizes v into row + column vectors)

For 7B model:
  Adam optimizer:      78 GB
  AdaFactor optimizer: ~50 GB  (saves ~28 GB)
```

AdaFactor stores v as outer product of row/column factors instead of full matrix. Used by T5, PaLM. Tradeoff: slightly worse convergence on some tasks.

## What Comes Next

Now that you understand where GPU memory goes, the next pods teach you how to reduce each component:
- Gradient checkpointing → reduce activations
- Gradient accumulation → simulate large batch without memory
- Data parallelism → train faster
- Tensor parallelism → split weights
- ZeRO → shard optimizer/gradients/weights
