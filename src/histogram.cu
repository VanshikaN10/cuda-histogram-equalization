// Version 1 here — naive, intentionally wrong, demonstrates race conditions
#include "histogram.cuh"

__global__ void histogram_naive(
    unsigned char* d_image,
    unsigned int* d_hist,
    int width,
    int height
) {
    // Figure out which pixel this thread is responsible for This is the fundamental CUDA indexing formula
    int pixel_index = blockIdx.x*blockDim.x+threadIdx.x;


    // Total number of pixels in the image
    int total_pixels = width*height;


    // Safety check — if this thread's pixel index is beyond the image size, do nothing and return because it doesnot exists
    // This happens because we launch more threads than pixels to keep block sizes clean (multiples of 32)
    if (pixel_index >= total_pixels) return;

    // Each pixel has 3 values — R, G, B stored one after another
    // So pixel 0 is at positions 0, 1, 2, Pixel 1 is at positions 3, 4, 5
    // Pixel N is at positions N*3, N*3+1, N*3+2
    int r = d_image[pixel_index * 3 + 0]; // Red value
    int g = d_image[pixel_index * 3 + 1]; // Green value
    int b = d_image[pixel_index * 3 + 2]; // Blue value

    // Now update the histogram bins, d_hist has 768 slots:
    // Slots 0-255   = Red histogram
    // Slots 256-511 = Green histogram
    // Slots 512-767 = Blue histogram

    // THIS IS THE RACE CONDITION
    // No atomics, no coordination
    // Multiple threads with same RGB values will collide here
    // Read-Add-Write is NOT atomic — threads interfere with each other
    d_hist[r]               ++;  // Red bin
    d_hist[256 + g]         ++;  // Green bin
    d_hist[512 + b]         ++;  // Blue bin
}

// Declaration of Version 2 kernel — atomic operations
// Correct result but slower due to global memory contention
__global__ void histogram_atomic(
    unsigned char* d_image,
    unsigned int* d_hist,
    int width,
    int height
) {
    // Exact same indexing as Version 1
    // Every thread still owns one pixel
    int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    // Exact same bounds check as Version 1
    if (pixel_index >= total_pixels) return;

    // Exact same RGB reading as Version 1
    // Image stored as flat array, each pixel takes 3 slots
    int r = d_image[pixel_index * 3 + 0];
    int g = d_image[pixel_index * 3 + 1];
    int b = d_image[pixel_index * 3 + 2];

    // THIS is the only difference from Version 1
    // atomicAdd guarantees read-add-write is uninterruptible
    // No two threads can corrupt each other's count &(ampersand) gives the memory address of that specific histogram slot
    atomicAdd(&d_hist[r],         1);  // Red bin
    atomicAdd(&d_hist[256 + g],   1);  // Green bin
    atomicAdd(&d_hist[512 + b],   1);  // Blue bin
}



// Declaration of Version 3 kernel — privatized shared memory histogram
// Correct and fast — contention reduced to one merge per block
__global__ void histogram_privatized(
    unsigned char* d_image,
    unsigned int* d_hist,
    int width,
    int height
) {
    // Each block gets its own private histogram in shared memory
    // __shared__ means this array lives in shared memory means private to this block, invisible to other blocks
    __shared__ unsigned int private_hist[TOTAL_BINS];

    // Step 1 — Initialize shared memory to zero
    // We have 768 slots but potentially 1024 threads per block
    // So each thread initializes one slot, some threads do extra
    // threadIdx.x gives each thread a unique starting slot
    int tid = threadIdx.x;
    for (int i = tid; i < TOTAL_BINS; i += blockDim.x) {
        private_hist[i] = 0;
    }

    // Wait for ALL threads in block to finish initializing before anyone starts accumulating
    __syncthreads();

    // Step 2 — Each thread reads its pixel and updates the block's PRIVATE histogram in shared memory
    int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (pixel_index < total_pixels) {
        // Read RGB values exactly like Version 1 and 2
        int r = d_image[pixel_index * 3 + 0];
        int g = d_image[pixel_index * 3 + 1];
        int b = d_image[pixel_index * 3 + 2];

        // Update private histogram in shared memory Still need atomicAdd here because threads within the same block can still have the same pixel value
        // But now it's hitting shared memory not global memory Shared memory atomics are much faster
        atomicAdd(&private_hist[r],         1);
        atomicAdd(&private_hist[256 + g],   1);
        atomicAdd(&private_hist[512 + b],   1);
    }


    // Wait for ALL threads to finish accumulating before anyone starts merging to global
    __syncthreads();

    // Step 3 — Merge this block's private histogram into the final global histogram
    for (int i = tid; i < TOTAL_BINS; i += blockDim.x) {
        atomicAdd(&d_hist[i], private_hist[i]);
    }
}