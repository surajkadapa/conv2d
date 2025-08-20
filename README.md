# Multi-threaded 2D Convolution Benchmark (C++)

## Overview
This project implements a **naive single-threaded** and a **parallel multi-threaded** 2D convolution in C++ from scratch with no external libraries like OpenCV or Eigen.  
The goal is to:
- Understand how convolution works at a low level
- Compare single-thread vs multi-thread performance
- Profile using tools like `perf`
- Analyze cache and memory behavior in CPU-bound workloads

---

## Problem Statement
2D convolution is a fundamental operation in image processing and neural networks.  
It involves sliding a small filter/kernel over a larger input matrix and computing weighted sums.

Given:
- **Input size:** `H × W`
- **Kernel size:** `K × K`
- **Stride:** `1` (fixed)
- **No padding**

We benchmark convolution for **large matrices** to highlight CPU performance bottlenecks and scaling behavior.

---

## Implementation Details

### **Single-threaded version (`conv_single.cpp`)**
- Straightforward nested loop implementation.
- Each output element is computed sequentially.
- Easy to understand, but slow for large matrices.

### **Multi-threaded version (`conv_multi.cpp`)**
- Parallelizes over output rows using `std::thread`.
- Work is evenly divided among threads.
- Each thread operates on independent output regions (no locking required).
- Number of threads is configurable.

---

## Hardware & Test Environment
- **CPU:** Intel® Core™ 5 210H (8 cores / 12 threads)
- **L1 Data Cache:** 32 KB per core
- **L2 Cache:** 2 MB per cluster
- **L3 Cache:** 12 MB shared
- **OS:** Ubuntu 24.04 (Kernel 6.14)
- **Compiler:** g++ 13.2.0 with `-O3 -march=native`

---

## Benchmark Configuration
- **Matrix size:** `2048 × 2048`
- **Kernel size:** `5 × 5`
- **Repetitions:** `3` (best time recorded)
- **Multi-thread test:** 8 threads

---

## Performance Results

| Version       | Time (ms) | Speedup vs Single | CPUs Utilized | Notable `perf` Observations |
|--------------|-----------|-------------------|--------------|-----------------------------|
| **Single-thread** | 27.98 ms  | 1.00×             | ~0.99        | Backend bound ~25%, ~3.47 G L1 loads/sec |
| **Multi-thread (8 threads)** | 9.75 ms   | **2.87×**         | ~2.24        | Higher LLC traffic, backend bound ~28% |

---

## Key Insights from Profiling

### **1. CPU Utilization**
- Single-thread version used ~1 core fully.
- Multi-threaded version averaged ~2.24 cores due to **memory bandwidth bottlenecks**.

### **2. Cache Behavior**
- Multi-threading reduced **L1 miss %** but increased **Last-Level Cache (LLC)** accesses.
- Indicates more distributed working sets but higher contention for shared cache.

### **3. Scaling Limitations**
- Ideal scaling with 8 threads would be ~8× faster.
- Real-world speedup is **2.87×** due to:
  - Memory bandwidth saturation
  - Cache miss penalties
  - Limited parallelism in the naive algorithm

---

## How to Build & Run

```bash
# Build
g++ -O3 -march=native conv_single.cpp -o conv_single
g++ -O3 -march=native -pthread conv_multi.cpp -o conv_multi

# Run Single-thread
./conv_single <input_height> <input_width> <kernel_size>

# Run Multi-thread
./conv_multi <input_height> <input_width> <kernel_size> <num_threads>

#Example
./conv_single 2048 2048 5
./conv_multi 2048 2048 5 8
```
