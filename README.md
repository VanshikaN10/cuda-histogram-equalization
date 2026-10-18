# CUDA Histogram Equalization Pipeline

GPU-accelerated RGB image contrast enhancement pipeline built in CUDA C++.

## What it does
Takes an RGB image and enhances contrast using histogram equalization —
redistributing pixel intensity values across the full 0-255 range per channel.
Implements three progressively optimized CUDA kernels benchmarked against
an OpenCV CPU baseline.

## Three kernel versions

| Version | Approach | Result |
|---------|----------|--------|
| Naive | Direct global memory increment | Incorrect — race conditions |
| Atomic | atomicAdd on global memory | Correct — contention bottleneck |
| Privatized | Per-block shared memory histogram | Correct — fast |

## Benchmark results
Tested on a 1024×734 RGB image on Tesla T4 GPU.

| Implementation | Time | Speedup vs CPU |
|----------------|------|----------------|
| CPU (OpenCV) | 3.70 ms | 1x |
| GPU Naive | 0.24 ms | — (incorrect) |
| GPU Atomic | 0.50 ms | 7.4x |
| GPU Privatized | 0.09 ms | **42x** |

## Key concepts demonstrated
- Race conditions in parallel histogram computation
- Atomic operations for correctness under contention
- Shared memory privatization to eliminate global memory bottleneck
- GPU memory hierarchy — global vs shared memory tradeoffs
- cudaEvent_t GPU-side benchmarking
- Parallel prefix sum for cumulative distribution function

## Tech stack
C++, CUDA, OpenCV, CMake — runs on any NVIDIA GPU

## How to run

### On Google Colab (recommended)
```python
!git clone https://github.com/VanshikaN10/cuda-histogram-equalization.git
%cd cuda-histogram-equalization
!apt-get install -y libopencv-dev
!mkdir build && cd build && cmake .. && make
!./build/histogram_eq
```

### Local (requires NVIDIA GPU + CUDA Toolkit)
```bash
git clone https://github.com/VanshikaN10/cuda-histogram-equalization.git
cd cuda-histogram-equalization
mkdir build && cd build
cmake ..
make
./histogram_eq
```

## Before / After
Input image placed in `images/input.jpg` — output saved to `images/output.jpg`.
For dramatic visual results use a dark or low-contrast input image.

## Project structure
```
src/
├── main.cu           — pipeline orchestration
├── histogram.cu/cuh  — three kernel versions
├── equalization.cu/cuh — prefix sum + pixel remap
└── benchmark.cu/cuh  — cudaEvent_t timing utilities
```