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

### Methodology

For instance, this is what the output looks like for a benchmark where I'm running 10 iterations for each algorithm, on an array size of $2^{30}$:

```
********************
** SCAN BENCHMARK **
********************

- Number of iterations: 10
- Size of POT array: 1073741824
- Size of NPOT array: 1073741821

[CPU/POT] Average scan() time: 268.988
[CPU/NPOT] Average scan() time: 287.848
[Naive/POT] Average scan() time: 1381.91
[Naive/NPOT] Average scan() time: 1382.35
[Efficient/POT] Average scan() time: 233.941
[Efficient/NPOT] Average scan() time: 239.228
[Thrust/POT] Average scan() time: 23.0701
[Thrust/NPOT] Executing scan(): 6 of 10...
```

### Graphs

### Analysis

### Miscellaneous: powers of scale

Just wanted to share some other fun stuff I encountered while testing.

I originally was testing with *much* smaller array sizes, like $2^4$ and $2^{12}$. When I tried increasing the array size past $2^{18}$, the program would instantly crash. I was really confused why at first, until I looked at the exception being thrown: *stack overflow*. Because I was using `std::array` for my input and output arrays, I was allocating too much stack memory and literally ran out. Switching to heap allocation solved the issue.

I then tried testing with array sizes from $2^{18}$ to $2^{30}$, incrementing by 4. This turned out to not be helpful at all; my numbers ranged from 0.068ms using CPU and $2^{18}$ to 1380.23ms using naive and $2^{30}$. Furthermore, my naive at $2^{26}$ ran in 75ms, so there was a ~18× difference between two adjacent data points. This would have translated to a *horrible* graph, so I adjusted the numbers to what I have now. 

Both of these experiences really left me with a newfound appreciation for exponents and the powers of two. It's *scary* how fast numbers can scale.

## Test output