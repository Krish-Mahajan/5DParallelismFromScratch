# Notebook 03: GPU Profiling

## Core Idea

Use `torch.profiler` to measure exactly where time goes during training — forward, backward, optimizer, data loading, individual kernels. This tells you whether you're compute-bound, memory-bound, or data-loading-bound, and what to optimize.

## torch.profiler Basics

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    with record_function("forward_pass"):
        output = model(x)
    with record_function("backward_pass"):
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

`record_function` creates custom labels in the profiler output. Without them, you'd only see low-level kernel names.

## Key Profiler Table Columns

| Column | What it means |
|--------|---------------|
| Self CPU / Self CUDA | Time in this op only (not sub-calls) |
| CPU total / CUDA total | Time including all sub-operations |
| # of Calls | How many times it ran |
| CPU Mem / CUDA Mem | Memory allocated/freed |

**Important:** Percentages can exceed 100% when mixing hierarchy levels (custom labels + kernel names) because of double-counting between parents and children.

## CUDA Async Caveat

CUDA operations are asynchronous — `loss.backward()` returns to CPU immediately, but GPU keeps working. This means:
- `record_function("backward_pass")` may show near-zero CUDA time (kernels attributed to autograd ops instead)
- Per-kernel breakdown (Part 2 table) is the reliable view
- Use `torch.cuda.synchronize()` for accurate wall-clock timing

## Training Phase Breakdown (Part 3)

Typical distribution for a CNN:
```
Forward Pass:      ~25-35% GPU time
Backward Pass:     ~45-55% GPU time (2× forward — grad_input + grad_weight)
Optimizer Step:    ~10-20% (Adam reads/writes m, v, weights)
Data Transfer:     ~1-10% (depends on DataLoader config)
```

## DataLoader Tuning (Part 4)

```
workers=0, no pin:    data transfer = 10.2% → GPU idle waiting for data
workers=2, pin=True:  data transfer = 5.2%  → overlap helps
workers=4, pin=True:  data transfer = 4.6%  → diminishing returns
```

- `num_workers > 0`: pre-loads next batch on CPU while GPU trains on current
- `pin_memory=True`: uses page-locked memory for faster CPU→GPU copies
- On H100 with small dataset (CIFAR-10): data loading is a minor bottleneck
- On larger datasets (ImageNet): data loading can be 30-50% without tuning

## CUDA Kernel Categories (Part 5)

Measured on CNN (CIFAR-10, H100):
```
Convolution:       54.5%  ← heavy compute (good GPU utilization)
Other (backward):  25.4%  ← autograd backward kernels
Pooling:            6.3%  ← memory-bound
Optimizer (Adam):   6.2%  ← memory-bound (reads/writes per-param states)
Matrix Multiply:    3.1%  ← FC layers
BatchNorm:          2.5%  ← memory-bound
Activation (ReLU):  1.9%  ← memory-bound (trivial math)
```

**Interpretation:**
- Convolution + MatMul dominate → **compute-bound** (healthy, GPU fully utilized)
- If Memory Transfer dominated → **data-loading bottleneck**
- If BatchNorm/Activation dominated → model too shallow, GPU underutilized

**What is a kernel?** One specific task sent to the GPU (e.g., "multiply these matrices", "apply ReLU to every element"). One Python line = many kernels launched sequentially.

## Batch Size vs GPU Utilization (Part 6)

Three charts:

**Throughput (samples/sec):** Rises steeply with batch size, then plateaus. The plateau = GPU is fully saturated.

**Step time (ms):** Grows roughly linearly. Double batch ≈ double step time.

**Memory (GB):** Grows linearly (activations ∝ batch_size). At some point → OOM.

```
Small batch (8-16):   low throughput, GPU mostly idle
Sweet spot (64-256):  peak throughput, GPU well utilized
Large batch (512+):   throughput plateaus, memory explodes
```

Optimal batch size = where throughput plateaus. Going bigger wastes memory without speed gain.

## Mixed Precision Profiling (Part 7)

**Results on H100 with small CNN:**
```
                FP32         AMP        Speedup   Memory saved
bs=64:         17314 s/s    13285 s/s    0.77×      21%
bs=128:        35005 s/s    25810 s/s    0.74×      28%
bs=256:        52124 s/s    46461 s/s    0.89×      35%
```

**AMP is SLOWER for this small model!** Why:
- CNN kernels are tiny → AMP overhead (autocast, GradScaler, dtype casting) > compute savings
- H100's FP32 is already very fast (67 TFLOPs)
- Small 32×32 feature maps don't tile efficiently onto Tensor Cores

**Memory savings are still real (21-35%)** — activations stored in FP16 regardless.

**When AMP helps vs hurts:**
```
Small CNN on H100 (32×32 images):     0.7-0.9× (slower! overhead dominates)
Large transformer (d=4096, seq=2048):  1.5-2.0× (faster! Tensor Cores shine)
```

AMP shines when: large matrices (4096×4096), compute-dominated workload, enough work to amortize the overhead.

**Interview answer:** "Mixed precision doesn't always help. For small models on fast GPUs, the overhead of dtype casting and GradScaler outweighs the compute benefit. It's most effective for large models with large matrix multiplications that fully utilize Tensor Cores."

## What Comes Next

Combining all profiling insights: identify bottlenecks → apply the right optimization (larger batch, mixed precision, more workers, checkpointing) based on whether you're compute-bound, memory-bound, or data-loading-bound.
