**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Charles Wang
  * [LinkedIn](https://linkedin.com/in/zwcharl)
  * [Personal website](https://charleszw.com)
* Tested on:
  * Windows 11 Pro (26100.4946)
  * Ryzen 5 7600X @ 4.7Ghz
  * 32 GB RAM
  * RTX 5060 Ti 16 GB (Studio Driver 580.97)

# CUDA Stream Compaction

This project implements multiple commonly used GPU algorithms, which are reduction, computing prefix sums (scan), and stream compaction. Stream compaction uses the scan algorithm under the hood, and one of my implementations for finding prefix sums uses a parallel reduction, so these algorithms are all building on each other.

The purpose of this project was to understand these algorithms in more detail, and explore how their implementations change when we parallelize them on the GPU. It also taught me more about how CUDA works and how my kernels interact with the physical NVIDIA hardware.

## Implementations

In order to explore potential performance differences, this project includes three different versions of the scan and compaction algorithms.

- [`cpu.cu`](stream_compaction/cpu.cu): these implementations run entirely on the CPU and are written in pure C++. They are single threaded by nature. In particular, the compaction algorithm was implemented both with and without using scan.
- [`naive.cu`](stream_compaction/naive.cu): the first implementation that utilizes CUDA. It is based on the naive algorithm described in [GPU Gems 3, Chapter 39.2.1](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).
- [`efficient.cu`](stream_compaction/efficient.cu): implementations of the scan and compaction algorithms which theoretically require less operations and therefore should run more efficiently. It is based on the work-efficient parallel scan algorithm described in [GPU Gems 3, Chapter 39.2.2](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda), and involves an "up-sweep" where we build up a balanced binary tree, and then a "down-sweep" where we calculate final terms using the node elements in the tree.

Below I demonstrate how my algorithms performed in benchmarking tests.

## Performance benchmarks

### Testing methodology

### Graphs

### Analysis
