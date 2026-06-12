# Notebook 01: Ring AllReduce Simulation

## Core Idea

AllReduce is the communication operation that averages gradients across all GPUs in data parallelism. Naive approach (gather to master) creates a bottleneck. Ring AllReduce distributes the work evenly — every GPU sends and receives the same amount.

## The Communication Problem

After backward pass, each GPU has its own local gradient (from its own mini-batch). All GPUs need the **average** of all gradients to take identical optimizer steps.

```
GPU 0: grad_0     ─┐
GPU 1: grad_1     ─┼── AllReduce ──→ every GPU gets: (grad_0 + grad_1 + grad_2 + grad_3) / 4
GPU 2: grad_2     ─┤
GPU 3: grad_3     ─┘
```

## Naive AllReduce (The Bad Approach)

**Algorithm:** Everything goes through GPU 0 (master).

```
Phase 1 — Gather: GPUs 1,2,3 send their gradients to GPU 0
Phase 2 — Average: GPU 0 computes mean
Phase 3 — Broadcast: GPU 0 sends result back to GPUs 1,2,3
```

**GPU 0 traffic:** `2 × (N-1) × gradient_size`

**The scaling problem:**
```
1M params (4 MB gradient):
  N=4:   GPU 0 handles 24 MB    (6× gradient)
  N=16:  GPU 0 handles 120 MB   (30× gradient)
  N=64:  GPU 0 handles 504 MB   (126× gradient)
```

GPU 0 becomes the bottleneck — all other GPUs sit idle waiting. Adding more GPUs makes it WORSE. This is why production systems never use naive AllReduce.

## The Simulation Infrastructure

```python
class SimulatedGPU:
    def __init__(self, gpu_id, gradient):
        self.data = gradient.copy()       # this GPU's gradient
        self.bytes_sent = 0               # track communication cost
        self.bytes_received = 0

    def send(self, chunk):
        self.bytes_sent += chunk.nbytes
        return chunk.copy()

    def receive(self, chunk):
        self.bytes_received += chunk.nbytes
        return chunk.copy()
```

Simulates multi-GPU communication on a single GPU. Tracks exactly how many bytes each GPU sends/receives to compare algorithms.

## Ring AllReduce — The Code

**Setup:** Split each GPU's gradient into N chunks:
```
GPU 0: [a,b,c,d,e,f,g,h] → [[a,b], [c,d], [e,f], [g,h]]
                              chunk0  chunk1  chunk2  chunk3
```

**Phase 1 — Scatter-Reduce (N-1 steps):**

Each GPU sends one chunk to its right neighbor, who **adds** it:

```python
for step in range(N - 1):
    for i in range(N):
        send_idx = (i - step) % N
        send_to = (i + 1) % N
        # Neighbor ADDS the received chunk to its own
        new_chunks[send_to][send_idx] = chunks[send_to][send_idx] + received
```

After N-1 steps, each GPU holds the **complete sum** of one chunk (different chunk per GPU).

**Phase 2 — Allgather (N-1 steps):**

Same ring passing, but **replace** instead of add:

```python
for step in range(N - 1):
    for i in range(N):
        send_idx = (i + 1 - step) % N
        send_to = (i + 1) % N
        # Neighbor REPLACES with the received chunk (it's already fully reduced)
        new_chunks[send_to][send_idx] = received
```

After N-1 steps, every GPU has ALL fully reduced chunks.

**Final:** Concatenate chunks and divide by N for the average.

**Key difference between phases:**
- Phase 1: `existing + received` (building the sum)
- Phase 2: `= received` (distributing the final result)

## Bandwidth Analysis

```
Naive:  master traffic = 2 × (N-1) × D → grows with N (bottleneck!)
Ring:   per-GPU traffic = 2 × (N-1)/N × D → approaches 2D regardless of N

100M params (400 MB gradient):
  N=4:    Ring = ~300 MB/GPU,   Naive master = 2.4 GB
  N=64:   Ring = ~394 MB/GPU,   Naive master = 50 GB (!)
  N=256:  Ring = ~399 MB/GPU,   Naive master = 200 GB (!!)
```

Ring per-GPU cost is nearly constant — adding GPUs doesn't increase communication time.

## Communication Time Formula

```
Ring time = 2 × (N-1) × (D/(N×B) + L)

D = gradient bytes
B = bandwidth (bytes/sec)
N = number of GPUs
L = per-step latency (seconds)
```

At large N, the `D/(N×B)` term shrinks (more GPUs = smaller chunks) but the `L` term accumulates (more steps × latency). For large gradients, bandwidth dominates. For tiny gradients, latency dominates.

## Gradient Bucketing (PyTorch DDP Optimization)

Instead of one giant AllReduce after backward, PyTorch groups parameters into **buckets** and starts AllReducing each bucket as soon as its gradients are ready.

```
Without bucketing:
  [full backward pass] → [full AllReduce]  (sequential)

With bucketing:
  [backward layer 12] → [start AllReduce bucket 1]
  [backward layer 11] → [AllReduce bucket 1 continues...]
  [backward layer 10] → [start AllReduce bucket 2]
  ...
  Communication overlaps with backward computation!
```

This hides most of the communication behind the backward pass — only the last bucket's AllReduce is fully "exposed."

## Ring vs Tree AllReduce

| | Ring | Tree |
|---|---|---|
| Steps | 2×(N-1) | 2×log₂(N) |
| Data per step | D/N | D/2 |
| Total bandwidth | 2×(N-1)/N × D ≈ 2D | log₂(N) × D |
| Best for | Large messages (bandwidth-optimal) | Small messages (fewer latency hits) |

Ring is bandwidth-optimal (less total data moved). Tree has fewer steps (less latency impact). Production NCCL uses ring for large tensors, tree for small ones.

## Latency vs Bandwidth Regimes

```
Ring time = 2 × (N-1) × (D/(N×B) + L)
                         ↑ bandwidth   ↑ latency
```

**Crossover point:** where bandwidth term = latency term → `D = N × B × L`

```
N=4:   crossover at 0.5 MB
N=8:   crossover at 1.0 MB
N=64:  crossover at 8.0 MB
```

- **Below crossover (small gradients):** latency dominates. More GPUs = more steps = slower. Ring isn't ideal.
- **Above crossover (large gradients):** bandwidth dominates. Ring is optimal. This is always the case for real models (gradients are 100s of MB).

For tiny tensors, tree-allreduce (fewer steps = less latency) beats ring. NCCL automatically picks the best algorithm.

## What Comes Next

Batch size experiments — how batch size affects convergence, throughput, and the compute-communication tradeoff.
