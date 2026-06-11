# 5D Training Parallelism From Scratch

Educational course on GPU programming and distributed training parallelism — from GPU architecture basics through all five parallelism dimensions (data, tensor, pipeline, sequence, expert).

## Structure

### Introduction
GPU fundamentals and overview of all parallelism strategies:
1. **GPU Architecture Basics** — cores, memory hierarchy, bandwidth
2. **Why GPUs for Deep Learning** — FLOPs breakdown, profiling, memory analysis
3. **Parallelism Strategies Overview** — data, tensor, pipeline, sequence, expert parallelism

### GPUMemory
Understanding exactly where GPU memory goes during training:
1. **Memory Anatomy** — weights, gradients, optimizer states (16 bytes/param)
2. **Activations & Mixed Precision** — Megatron-LM formula, autocast, GradScaler
3. **Will It Fit?** — memory calculator, max batch size predictor, compatibility matrix

### SingleGPUToMultiGPU
From single-GPU optimizations to distributed training:
1. **Activation Checkpointing** — recompute activations during backward (trade compute for memory)
2. **Gradient Accumulation** — simulate large batches without extra memory
3. **Data Parallelism** — replicate model, split data, all-reduce gradients

### (Upcoming)
- Tensor Parallelism — column/row parallel, Megatron-LM patterns
- Pipeline Parallelism — 1F1B scheduling, micro-batches
- ZeRO Optimizer — shard optimizer/gradients/weights across GPUs
- Sequence Parallelism — ring attention for long contexts
- Expert Parallelism — MoE routing and load balancing

## Hardware

Notebooks run on a P5.48xlarge instance (8x NVIDIA H100 80GB GPUs) via VS Code remote Jupyter kernel.

## Summary Documents

Each notebook has a corresponding summary markdown file with key concepts, formulas, intuitive explanations, and interview-ready answers. Deep-dive documents (e.g., `tensor_parallelism_deep_dive.md`) cover topics in extra detail.
