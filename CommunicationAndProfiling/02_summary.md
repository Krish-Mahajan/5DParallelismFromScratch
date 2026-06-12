# Notebook 02: Batch Size Experiments

## Core Idea

Batch size affects convergence, throughput, and communication overhead. Larger batches make GPUs efficient but converge worse without learning rate adjustments. Understanding this tradeoff is essential for distributed training.

## The Batch Size Tradeoff

| Metric | Small Batch (64) | Large Batch (4096) |
|--------|-----------------|-------------------|
| Gradient noise | High (noisy but exploratory) | Low (accurate but cautious) |
| Steps per epoch | Many | Few |
| GPU utilization | Low (underutilized) | High (saturated) |
| Throughput (samples/sec) | Low | High |
| Convergence (same LR) | Fast per epoch | Slow per epoch |

## Experiment 1: Fixed LR, Varying Batch Size

**Setup:** Train same model (CNN on CIFAR-10) with LR=0.01, batch sizes [64, 256, 1024, 4096].

**Results (4 plots):**

**Loss per step:** Smaller batches drop loss faster per step (more frequent updates, each noisy but useful).

**Loss per epoch:** bs=64 achieves lowest loss after 10 epochs. bs=4096 converges worst — the LR is effectively "too small" for the smooth gradients.

**Test accuracy:** Same pattern — bs=64 highest accuracy, bs=4096 lowest (underfitting, hasn't converged yet).

**Throughput:** bs=4096 processes the most samples/sec (GPU fully saturated with large matmuls). bs=64 is slowest (tiny batches underutilize GPU).

**Key insight:** Bigger batch = faster per sample but worse convergence with same LR. The learning rate needs to scale with batch size.

## Why Fixed LR Fails for Large Batches

With batch=64 and LR=0.01:
- Gradient is noisy (high variance) but you take a step of size 0.01
- Effectively: "take a risky but frequent step"

With batch=4096 and LR=0.01:
- Gradient is accurate (low variance) but you take the SAME size step (0.01)
- Effectively: "take a cautious step with perfect information — wasteful!"
- The gradient is 64× more accurate but the step size doesn't increase to match

The fix: scale LR proportionally so the step size matches the gradient quality.

## Learning Rate Scaling Rules (Experiment 2)

**Linear scaling (Goyal et al., 2017):**
```
lr_new = lr_base × (bs_new / bs_base)

Example: base bs=64, lr=0.01
  bs=256:  lr = 0.01 × 4 = 0.04
  bs=4096: lr = 0.01 × 64 = 0.64
```

**Square root scaling (Hoffer et al., 2017):**
```
lr_new = lr_base × √(bs_new / bs_base)

Example: base bs=64, lr=0.01
  bs=256:  lr = 0.01 × 2 = 0.02
  bs=4096: lr = 0.01 × 8 = 0.08
```

Linear is more aggressive (matches the noise reduction exactly). Square root is more conservative (works better in practice for very large batches where linear can be unstable).

### Experiment 2 Results (3 charts)

**Chart 1 — Loss per Epoch:** With proper LR scaling, all batch sizes converge to similar final loss. Without scaling (Part 3), large batches lagged behind.

**Chart 2 — Accuracy per Epoch:** Scaled configs reach similar accuracy as the base. Linear scaling matches baseline closely. Sqrt is slightly slower but more stable.

**Chart 3 — Accuracy vs Wall-Clock Time (the real-world comparison):**
```
bs=64:   best accuracy per epoch, BUT slowest in wall-clock (low GPU utilization)
bs=1024: same accuracy, reached FASTER in wall-clock (high throughput)
```

This is the payoff: "increase batch 16× + scale LR 16× = same convergence in 16× less wall-clock time (given 16 GPUs)."

### Linear vs Sqrt Scaling — When to Use Which

| Scaling | Formula | Risk | Best for |
|---------|---------|------|----------|
| Linear | lr × (bs/bs_base) | Can be unstable at very large bs | bs up to ~8K with warmup |
| Sqrt | lr × √(bs/bs_base) | Slower convergence | Very large bs (16K+), stability matters |

In practice: start with linear + warmup. If training is unstable, fall back to sqrt.

## Warmup for Large Batches (Experiment 3)

Large LR + random initial weights = unstable early training. Solution: start with tiny LR, ramp up linearly over first few hundred steps.

**Config:** bs=1024, lr=0.16 (linear scaled), warmup = [0, 50, 200, 500] steps.

**Chart 1 — Early Training Loss (first 300 steps):**
```
No warmup:        loss spikes wildly in first 50 steps → chaotic, may never recover
50-step warmup:   slight instability, then settles
200-step warmup:  smooth, loss drops steadily from the start
500-step warmup:  smoothest start (diminishing returns vs 200)
```

Why no-warmup fails: full LR=0.16 × unreliable gradients (from random weights) = catastrophic early updates → loss explodes.

**Chart 2 — Accuracy per Epoch:**
```
No warmup:    lowest final accuracy (permanently damaged by early instability)
200-step:     near-optimal (enough warmup to stabilize)
500-step:     similar to 200 (diminishing returns)
```

**Key insight:** Early damage from no-warmup is permanent — even after 10 epochs the model can't catch up. Warmup of ~200 steps (5-10% of first epoch) is enough. Every production recipe uses it.

**Production standard:** GPT/LLaMA use warmup_steps=2000 for ~300K total steps.

## Gradient Noise Analysis (Experiment 4)

**Gradient noise = variance of gradient estimate across different mini-batches.**

Measured empirically: sample 20 mini-batches at each batch size, compute gradient for each, measure how much they differ.

**Chart 1 — Gradient Variance vs Batch Size (log-log):**
```
bs=32:   high variance (each mini-batch gives very different gradients)
bs=2048: low variance (gradients are consistent)

Relationship: Var(gradient) ∝ 1/batch_size  (straight line on log-log)
```

**Chart 2 — Mean Gradient Norm:**
Stays roughly constant — batch size changes noise, not the average direction.

**Chart 3 — Signal-to-Noise Ratio (SNR = mean_norm / std):**
```
SNR ∝ √batch_size

Higher SNR = gradient is more "trustworthy" → can take bigger steps
```

**Why this explains LR scaling:**
- Variance drops as 1/B → std drops as 1/√B → SNR grows as √B
- Linear scaling (LR ∝ B): slightly aggressive — assumes noise drops linearly
- Sqrt scaling (LR ∝ √B): matches the actual noise reduction exactly
- In practice, linear works up to ~8K batch with warmup, sqrt is safer beyond

This is the fundamental reason batch size matters:
- High noise → need small steps (low LR) to avoid overshooting
- Low noise → can take big steps (high LR) safely
- Scaling LR with batch size matches step size to noise level

## Distributed Training Simulation (Experiment 5)

Simulating N GPUs = multiplying effective batch size by N:

```
1 GPU:  bs=256, lr=0.01
4 GPUs: effective bs=1024, lr scaled to 0.04 (linear) or 0.02 (sqrt)
8 GPUs: effective bs=2048, lr scaled to 0.08 (linear) or 0.028 (sqrt)
```

Linear speedup is achievable if:
1. Scale LR appropriately
2. Use warmup for the first few hundred steps
3. Communication overhead (AllReduce) doesn't dominate

## The Complete Recipe for Large-Batch Training

```
1. Pick base config: bs_base, lr_base (known to work on 1 GPU)
2. Scale batch: bs_new = bs_base × N_GPUs
3. Scale LR: lr_new = lr_base × (bs_new / bs_base)  [linear scaling]
4. Add warmup: ramp LR from 0 to lr_new over first 5-10% of steps
5. Optionally: reduce LR if training is unstable (fall back to sqrt scaling)
```

## What Comes Next

GPU profiling — measuring where time actually goes during training (compute vs memory vs communication).
