# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational course on GPU programming and distributed training parallelism (5D parallelism), structured as a series of Jupyter notebooks organized by chapter. The notebooks are designed to run on the **P5.48xlarge instance** (8x H100 GPUs) in ap-south-1.

The course is part of the "Vizuara AI Pods | GPU Programming Course" and progresses from GPU architecture basics through advanced parallelism strategies.

## Structure

- `Chapter1/` — Intro to GPUs and GPU Parallelism (fundamentals, benchmarking, memory)
- Subsequent chapters will cover deeper parallelism dimensions (data, tensor, pipeline, expert, sequence parallelism)

## Development Setup

**Primary workflow: VS Code + Remote Jupyter Kernel**

Notebooks are edited locally in VS Code and executed on the P5 via a remote kernel connection. Outputs are saved in the local `.ipynb` file on save — no sync needed for day-to-day work.

1. Start Jupyter on P5 (SSH in first with `ssh-p5-ap-south-1`):
   ```bash
   source /opt/pytorch/bin/activate
   cd ~/mahajak-workspace/5DTrainingParallelism
   jupyter notebook --no-browser --ip=0.0.0.0 --port=8889
   ```

2. Port forward from Mac (separate terminal):
   ```bash
   export $(ada credentials print --provider conduit --account 199086640399 --role magnus-training-users --format env | xargs) && \
   aws ec2-instance-connect send-ssh-public-key \
     --instance-id i-0d0a09580b9f3a0b7 \
     --instance-os-user ubuntu \
     --ssh-public-key file:///tmp/temp_ec2_key.pub \
     --region ap-south-1 && \
   ssh -i /tmp/temp_ec2_key -L 8889:localhost:8889 -N ubuntu@65.2.170.61
   ```

3. In VS Code: open notebook → Select Kernel → Existing Jupyter Server → `http://127.0.0.1:8889`

Note: Port 8888 is typically in use — use **8889**.

**SSH key regeneration (after Mac reboot):**
```bash
ssh-keygen -t ed25519 -f /tmp/temp_ec2_key -N "" -q
```

## Collaboration Workflow with Claude

- **Explain in chunks:** When walking through code, go one chunk at a time and wait for "next" or "yes" before continuing
- **Summary docs:** Each chapter/notebook should have a `XX_summary.md` with key concepts, formulas, intuitive explanations, and interview-ready answers
- **Push to GitHub:** After finishing a notebook/summary, commit and push (repo: `Krish-Mahajan/5DParallelismFromScratch`)
- **Notebook reading:** Claude can read local `.ipynb` files directly after the user saves in VS Code (outputs are stored locally)
- **Sync for file transfers:** Only needed when pushing new files to P5 or pulling files edited directly on P5

## Notebook Conventions

- Each notebook includes audio narration cells (using `#@title` annotations for Colab form fields) that download and play MP3 segments from Google Drive
- Code cells alternate between narration/audio cells, markdown explanations, and executable code
- Exercises are marked with `# TODO` comments for learners to complete
- All notebooks assume `torch`, `numpy`, and `matplotlib` available

## Key Technical Context

- Target hardware: P5.48xlarge with 8x H100 80GB GPUs
- Remote workspace: `/home/ubuntu/mahajak-workspace/5DTrainingParallelism/` on the P5 instance
- The P5 instance is shared with other teams — only work inside the mahajak-workspace folder
- Benchmarks use `torch.cuda.synchronize()` for accurate GPU timing
- Memory measurements use `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`
- The course covers FP32, FP16, and BF16 precision comparisons
- Arithmetic intensity (FLOPs/byte) is a recurring concept for understanding compute vs memory bottlenecks
