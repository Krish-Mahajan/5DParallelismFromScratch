# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational course on GPU programming and distributed training parallelism (5D parallelism), structured as a series of Jupyter notebooks organized by chapter. The notebooks are designed to run in **Google Colab** with GPU runtimes (T4/A100).

The course is part of the "Vizuara AI Pods | GPU Programming Course" and progresses from GPU architecture basics through advanced parallelism strategies.

## Structure

- `Chapter1/` — Intro to GPUs and GPU Parallelism (fundamentals, benchmarking, memory)
- Subsequent chapters will cover deeper parallelism dimensions (data, tensor, pipeline, expert, sequence parallelism)

## Notebook Conventions

- Each notebook includes audio narration cells (using `#@title` annotations for Colab form fields) that download and play MP3 segments from Google Drive
- Code cells alternate between narration/audio cells, markdown explanations, and executable code
- Exercises are marked with `# TODO` comments for learners to complete
- All notebooks assume a Colab environment with `torch`, `numpy`, and `matplotlib` available via pip

## Development

There is no local build system. Notebooks are authored and tested directly in Google Colab.

To run locally (if needed):
```bash
pip install torch numpy matplotlib jupyter
jupyter notebook
```

## Key Technical Context

- Benchmarks use `torch.cuda.synchronize()` for accurate GPU timing
- Memory measurements use `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`
- The course covers FP32, FP16, and BF16 precision comparisons
- Arithmetic intensity (FLOPs/byte) is a recurring concept for understanding compute vs memory bottlenecks
