# Notebook 03: DeepSpeed ZeRO

## Core Idea

DeepSpeed is a production library that implements ZeRO with one JSON config and 3 lines of code change. It handles all the sharding, communication, and memory management behind the scenes.

## DeepSpeed Configuration

All ZeRO behavior is controlled by a JSON dictionary:

```json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": true},
    "zero_optimization": {"stage": 1},
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}}
}
```

**Batch size relationship:** `train_batch_size = micro_batch × accumulation_steps × num_GPUs`

### Stage-Specific Config

**Stage 1** — minimal (one field):
```json
"zero_optimization": {"stage": 1}
```

**Stage 2** — adds communication optimizations:
```json
"zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,   // AllGather updated weights after step
    "reduce_scatter": true,          // ReduceScatter for gradients (not AllReduce)
    "overlap_comm": true             // overlap communication with backward (bucketing)
}
```

**Stage 3** — adds memory management for weight sharding:
```json
"zero_optimization": {
    "stage": 3,
    "param_persistence_threshold": 1e5,   // small params stay replicated (not worth sharding)
    "max_live_parameters": 1e9,            // max params gathered at once (peak memory control)
    "reduce_bucket_size": 5e5,             // bucket size for ReduceScatter
    "prefetch_bucket_size": 5e5            // prefetch next layer while computing current
}
```

## DeepSpeed API — 3 Lines Replace Your Training Loop

```python
# Standard PyTorch:                    # DeepSpeed:
model = Model()                         model = Model()
optimizer = Adam(model.params())        model_engine, _, _, _ = deepspeed.initialize(
                                            model=model, config=ds_config)

output = model(x)                       output = model_engine(x)
loss.backward()                         model_engine.backward(loss)
optimizer.step()                        model_engine.step()
optimizer.zero_grad()                   # (handled automatically)
```

`deepspeed.initialize()` wraps the model with ZeRO hooks and creates the optimizer internally.

## What DeepSpeed Does Behind the Scenes

```
Stage 1:
  .backward(loss): loss.backward() + AllReduce gradients
  .step():         update this rank's optimizer shard + AllGather weights

Stage 2:
  .backward(loss): loss.backward() + ReduceScatter gradients (bucketed, overlapped)
  .step():         update this rank's shard + AllGather weights

Stage 3:
  model_engine(x): AllGather weights layer-by-layer during forward
  .backward(loss): AllGather weights + ReduceScatter gradients
  .step():         update this rank's shard only
```

All complexity hidden behind the same 3 API calls.

## Single-GPU Environment Variables

DeepSpeed requires distributed env vars even on 1 GPU:
```python
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
```

## Important: Single GPU Limitations

On world_size=1, all ZeRO stages show similar memory because there's only 1 rank (nothing to shard across). Real savings appear with N>1 GPUs where each rank stores 1/N.

## ZeRO-Offload — CPU/NVMe Offloading

For extreme memory savings, offload optimizer states and/or parameters to CPU:

```json
"zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true}
}
```

**What happens with offload:**
```
Forward:
  Copy weight shard CPU → GPU → AllGather → compute → free GPU weights

Backward:
  AllGather from CPU → GPU → compute backward → ReduceScatter → send grad to CPU

Optimizer (runs on CPU!):
  CPU updates master weights with Adam (m, v are on CPU too)
  Updated weights stay on CPU until next forward
```

**The tradeoff:**
```
Without offload: fast (everything on GPU), limited by 80 GB
With offload:    slow (CPU↔GPU transfers), but ~unlimited memory (CPU has 100s of GB)
                 Typically 2-5× slower
```

`pin_memory: true` = page-locked CPU memory for faster CPU↔GPU DMA transfers.

**When to use offload:**
- Model doesn't fit even with ZeRO-3 across available GPUs
- Few GPUs but lots of CPU RAM (e.g., 1-2 consumer GPUs)
- Willing to trade speed for ability to train at all
- NOT needed on 8×H100 (plenty of GPU memory)

## Decision Matrix: Which Stage to Use

```
Model fits on 1 GPU?
  YES → Standard DP (no ZeRO needed)
  NO  → Does it fit with ZeRO-1?
         YES → ZeRO-1 (simplest, same communication as DP)
         NO  → Does it fit with ZeRO-2?
                YES → ZeRO-2 (recommended default — free memory, same comm)
                NO  → ZeRO-3 (1.5× comm, but linear memory scaling)
                       Still doesn't fit? → ZeRO-3 + CPU offload
```

## Quick Reference Cheat Sheet

```
# Launch DeepSpeed training:
deepspeed train.py --deepspeed_config ds_config.json

# In your training script:
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters()
)

# Training step:
loss = model_engine(batch)
model_engine.backward(loss)
model_engine.step()

# Save checkpoint:
model_engine.save_checkpoint(save_dir, tag)

# Load checkpoint:
model_engine.load_checkpoint(load_dir, tag)
```

## DeepSpeed ZeRO Quick Reference

```
STAGE 1 - Partition optimizer states
  Config:  {"stage": 1}
  Memory:  4*psi + 12*psi/N bytes per GPU
  Comm:    Same as standard DP (1.0x)
  Use when: Model fits but you want more headroom

STAGE 2 - Partition optimizer + gradients
  Config:  {"stage": 2, "reduce_scatter": true, "overlap_comm": true}
  Memory:  2*psi + 14*psi/N bytes per GPU
  Comm:    Same as standard DP (1.0x)
  Use when: Sweet spot for most training runs

STAGE 3 - Partition everything
  Config:  {"stage": 3, "param_persistence_threshold": 1e5}
  Memory:  16*psi/N bytes per GPU
  Comm:    1.5x standard DP
  Use when: Weights don't fit on one GPU

OFFLOAD - Move states to CPU/NVMe
  Config:  {"stage": 3, "offload_optimizer": {"device": "cpu"}}
  Memory:  Minimal GPU usage
  Comm:    CPU<->GPU transfer overhead
  Use when: Last resort, willing to be 2-5× slower

RULE OF THUMB:
  Start with ZeRO-2. Only move to ZeRO-3 if OOM.
  Only use offload as a last resort.
```

## Example: 1B Model on 4 A100-80GB GPUs

```
                 Memory/GPU    Fits?    Headroom for activations
Standard DP:     14.9 GB       YES      65 GB
ZeRO-1:          6.5 GB       YES      73 GB
ZeRO-2:          5.1 GB       YES      75 GB
ZeRO-3:          3.7 GB       YES      76 GB
```

For 1B model, even DP fits. ZeRO's value = extra headroom for larger batches/sequences.

## What Comes Next

Tensor parallelism — splitting individual weight matrices across GPUs for models where even a single layer doesn't fit on one GPU.
