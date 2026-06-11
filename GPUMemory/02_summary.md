# Notebook 02: Activation Memory and Mixed Precision Training

## Core Idea

Activations are intermediate values saved during forward pass for backward. Unlike weights/optimizer (fixed per model), activations scale with **batch_size × seq_len**. Mixed precision halves activation memory by storing them in fp16 instead of fp32.

## Why Activations Are Stored

During forward pass, each layer produces an output. Backward needs these outputs to compute gradients (chain rule). So PyTorch saves them in memory until backward runs through that layer.

**Textbook analogy:** Taking notes at each chapter so you can reference them when solving the problem set at the end. More chapters (layers) × more examples (batch) = more notes (memory).

## The Megatron-LM Formula

$$M_{\text{activations}} \approx L \times B \times S \times H \times \left(34 + 5 \cdot \frac{A \cdot S}{H}\right) \text{ bytes}$$

Where:
- L = number of layers
- B = batch size
- S = sequence length
- H = hidden dimension
- A = number of attention heads

**What each term stores:**

```
34 coefficient covers (per layer):
  Layer norm outputs:    B × S × H (×2)
  Q, K, V projections:  3 × B × S × H
  Attention output:      B × S × H
  FFN intermediate:      B × S × 4H
  FFN output:            B × S × H

5 × A×S/H covers:
  Attention scores:      B × A × S × S   ← QUADRATIC in seq_len!
```

**Simplified (moderate sequence lengths):**

$$M_{\text{activations}} \approx L \times B \times S \times H \times 34 \text{ bytes}$$

**When the quadratic term matters:**
```
LLaMA-7B, S=512:   A×S/H = 32×512/4096 = 4     → coefficient = 34 + 20 = 54
LLaMA-7B, S=2048:  A×S/H = 32×2048/4096 = 16   → coefficient = 34 + 80 = 114
LLaMA-7B, S=8192:  A×S/H = 32×8192/4096 = 64   → coefficient = 34 + 320 = 354 ← dominates!
```

## Activation Memory Scaling

| Dimension | Scaling | Implication |
|-----------|---------|-------------|
| Batch size (B) | Linear | **First knob to turn when OOM** |
| Sequence length (S) | ~Linear + S² (attention) | Long sequences blow up |
| Hidden dim (H) | Linear | Fixed by architecture |
| Number of layers (L) | Linear | Fixed by architecture |

**Activation memory for LLaMA-7B (S=2048):**
```
B=1:   ~15 GB
B=2:   ~30 GB
B=4:   ~60 GB
B=8:   ~120 GB
B=16:  ~240 GB
```

## Empirical Measurement Pattern

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model = Model().to(device)
mem_model = torch.cuda.memory_allocated()       # just weights

x = torch.randn(B, S, H, device=device)
output = model(x)                                # forward saves activations
mem_forward = torch.cuda.memory_allocated()

activation_memory = mem_forward - mem_model - input_size
```

## Mixed Precision Training

**The idea:** Forward/backward in fp16 (fast, half memory), optimizer in fp32 (stable).

**What saves memory:**
- Activations stored in fp16 (2 bytes) instead of fp32 (4 bytes) → **halves activation memory**
- Gradients in fp16 → half gradient memory
- Weights/optimizer: unchanged (still need fp32 master + m + v)

**The code pattern:**
```python
with autocast(dtype=torch.float16):        # forward in fp16
    output = model(input_ids)
    loss = cross_entropy(output, targets)

scaler.scale(loss).backward()              # scale gradients to prevent underflow
scaler.step(optimizer)                     # unscale + step (skip if overflow)
scaler.update()                            # adjust scale factor
```

**What `autocast` does:** Matmuls run in fp16 (fast on Tensor Cores). Precision-sensitive ops (softmax, layernorm, loss) stay in fp32 automatically.

**What `GradScaler` does:**
1. Multiplies loss by ~65536 before backward → gradients are large enough to not underflow in fp16
2. Divides gradients back down before optimizer step
3. If inf/NaN detected (overflow), skips the step and halves the scale factor
4. If clean, gradually increases the scale factor (more aggressive)

**Why it's needed:** fp16 smallest value ≈ 0.00006. Without scaling, tiny gradients become zero → training breaks.

**Results:**
```
Memory savings: 30-50% peak memory
Speed:          1.5-2× faster (fp16 matmuls on Tensor Cores)
Quality:        Nearly identical loss curves
```

## Decision Framework

```
Small batch (B=1-2):
  → Optimizer states dominate memory (~75%)
  → Optimization: ZeRO (distribute optimizer across GPUs)

Large batch (B=8+):
  → Activations dominate memory
  → Optimization: gradient checkpointing (recompute instead of store)
                  OR mixed precision (halve activation size)
```

## Complete Training Memory Formula

```
M_total = Weights + Gradients + Optimizer + Activations
        = 2P + 2P + 12P + L×B×S×H×34
        = 16P + L×B×S×H×34 bytes

(P = parameters, mixed precision + Adam)
```

## What Comes Next

"Will It Fit?" calculator — takes any model config and tells you if it fits on your GPU, with recommendations for what to change if not.
