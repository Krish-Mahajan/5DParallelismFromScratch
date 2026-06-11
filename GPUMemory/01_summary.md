# Notebook 01: Memory Anatomy — Weights, Gradients, and Optimizer States

## Core Idea

Training a model requires far more memory than just the weights. A 7B model's weights are ~14 GB in fp16, but training needs ~112 GB. The extra memory goes to gradients, optimizer states, and a master copy of weights.

## Key Formula

For mixed precision training with Adam, **16 bytes per parameter**:

| Component | Bytes per Param | For 7B model |
|-----------|----------------|--------------|
| Weights (fp16) | 2 | 13 GB |
| Gradients (fp16) | 2 | 13 GB |
| fp32 Master Weights | 4 | 26 GB |
| Adam 1st moment m (fp32) | 4 | 26 GB |
| Adam 2nd moment v (fp32) | 4 | 26 GB |
| **Total** | **16** | **104 GB** |

This is **8× the fp16 weight memory** — and doesn't include activations.

## The Business Analogy

- **Weights** = inventory (the core asset you optimize)
- **Gradients** = daily sales reports (which direction to adjust)
- **Optimizer states** = accountant's records (tracks trends and volatility for smarter adjustments)

The accountant's records (optimizer) are often larger than the inventory itself.

## Component 1: Model Weights

A lookup table — each parameter is a number stored in some floating-point format:

```
fp32: 4 bytes per param  (full precision)
fp16: 2 bytes per param  (half precision)
bf16: 2 bytes per param  (half precision, same range as fp32)
```

**bf16 vs fp16:** Both are 2 bytes. bf16 has the same range as fp32 (8-bit exponent) but less precision. For deep learning, range matters more than precision — prevents overflow/underflow. Preferred on hardware that supports it (A100, H100).

**Weight memory for common models:**

```
GPT-2 (124M):   fp32 = 0.46 GB,  fp16 = 0.23 GB
LLaMA-7B:       fp32 = 26.08 GB, fp16 = 13.04 GB
LLaMA-13B:      fp32 = 48.43 GB, fp16 = 24.21 GB
LLaMA-70B:      fp32 = 260.77 GB, fp16 = 130.39 GB
```

## Component 2: Gradients

During backward pass, PyTorch creates a gradient tensor for every trainable parameter. The gradient has **exactly the same shape** as the weight.

```
embed.weight:         [32000, 768]   grad: [32000, 768]   ← same!
attn.in_proj_weight:  [2304, 768]    grad: [2304, 768]    ← same!
ffn.0.weight:         [3072, 768]    grad: [3072, 768]    ← same!
```

**Gradient memory = weight memory.** 1:1 ratio.

In fp32 training: gradients cost another 4 bytes/param.
In mixed precision: gradients in fp16 = 2 bytes/param.

**Why gradients match weight shape:** The gradient of a loss with respect to a weight matrix W has the same dimensions as W — each entry tells "how much does the loss change if I nudge this specific weight?"

**The shapes in a GPT-2 transformer (hidden_dim=768):**

| Parameter | Shape | What it is |
|-----------|-------|-----------|
| embed.weight | [32000, 768] | Token embedding (vocab × d_model) |
| attn.in_proj_weight | [2304, 768] | Q,K,V packed (3×768=2304 × d_model) |
| attn.out_proj.weight | [768, 768] | Output projection (d_model × d_model) |
| ffn.0.weight | [3072, 768] | FFN expand (4×768=3072 × d_model) |
| ffn.2.weight | [768, 3072] | FFN compress (d_model × 4×768=3072) |
| head.weight | [32000, 768] | Output head (vocab × d_model) |

## Component 3: Optimizer States (Adam)

Adam stores **two extra tensors** per parameter, both in fp32:
- **First moment (m):** exponential moving average of gradients (direction/momentum)
- **Second moment (v):** exponential moving average of squared gradients (magnitude/variance)

```python
state['exp_avg']     → shape same as weight, dtype fp32  (m)
state['exp_avg_sq']  → shape same as weight, dtype fp32  (v)
```

**Why always fp32?** Small gradient values would underflow in fp16. The optimizer needs full precision to accumulate tiny updates over thousands of steps without losing information.

**Memory cost:**
```
Adam: 2 tensors × 4 bytes/param = 8 bytes/param for optimizer states
+ fp32 master weights (mixed precision): 4 bytes/param
Total optimizer overhead: 12 bytes/param
```

For a 7B model: 12 × 7B = 84 GB just for optimizer states + master weights.

**Optimizer states are the LARGEST memory consumer** (~75% of non-activation memory).

## SGD vs Adam Memory

| Optimizer | State tensors per param | Bytes/param | For 7B model |
|-----------|------------------------|-------------|--------------|
| SGD + momentum | 1 (momentum buffer, fp32) | 4 | 26 GB |
| Adam (mixed precision) | 3 (master + m + v, all fp32) | 12 | 78 GB |

Adam uses 3× more optimizer memory than SGD. Some people use SGD for very large models despite worse convergence — saves ~52 GB for 7B.

## Empirical Verification (GPU Memory at Each Stage)

Measured on a TinyTransformer (~124M params, fp32):

```
After loading model:     0.50 GB  ← weights only
After forward pass:      0.52 GB  ← + activations (small batch)
After backward pass:     1.00 GB  ← + gradients (same size as weights)
After optimizer step:    2.00 GB  ← + optimizer states (2× weight size for m,v)
```

Each stage adds memory. Optimizer step is the biggest jump — that's where m and v are allocated.

## Scaling Law

Training memory scales linearly with parameters (excl. activations):

```
fp32 training:    16 bytes/param  (4+4+8)
mixed precision:  16 bytes/param  (2+2+12)
fp16 training:    12 bytes/param  (2+2+8, no master copy)
```

Mixed and fp32 cost the same total (16 bytes/param) — mixed just redistributes it differently (less for weights/grads, more for optimizer with master copy).

## GPU Fit Check (excl. activations)

```
GPT-2 (124M):  ~1.8 GB   → fits on any GPU
LLaMA-7B:      ~104 GB   → exceeds A100 80GB!
LLaMA-13B:     ~194 GB   → needs 3+ A100s
LLaMA-70B:     ~1.04 TB  → needs 14+ A100s
```

**Key insight:** Even before activations, a 7B model doesn't fit on a single GPU for training. This is why ZeRO optimizer partitioning (distributing m, v across GPUs) is so impactful — it targets the largest component.

## What Comes Next

Notebook 02: Activations — the fourth memory component that scales with batch_size × seq_len.
