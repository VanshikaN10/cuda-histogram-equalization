// Contains two kernels that complete the equalization pipeline:
// 1. prefix_sum  — converts histogram to cumulative distribution
// 2. remap_pixels — uses CDF to remap every pixel to new value

#include "equalization.cuh"

__global__ void prefix_sum(
    unsigned int* d_hist,
    int num_bins
) {
    // Load all histogram values into shared memory
    // We use one block for this kernel because num_bins is only 768
    // Small enough to fit entirely in shared memory and process simply
    extern __shared__ unsigned int temp[];

    int tid = threadIdx.x;

    // Each thread loads one bin from global memory into shared memory
    if (tid < num_bins) {
        temp[tid] = d_hist[tid];
    }

    // Wait for all threads to finish loading
    __syncthreads();

    // Compute prefix sum in shared memory
    // This is a simple sequential scan — works fine for 768 values
    // Thread 0 does all the work, other threads wait
    if (tid == 0) {
        for (int i = 1; i < num_bins; i++) {
            temp[i] += temp[i - 1];
        }
    }

    // Wait for thread 0 to finish computing
    __syncthreads();

    // Write results back to global memory
    if (tid < num_bins) {
        d_hist[tid] = temp[tid];
    }
}

__global__ void remap_pixels(
    unsigned char* d_image,
    unsigned int* d_hist,
    int width,
    int height,
    int total_pixels
) {
    // Same indexing as histogram kernels
    // Every thread handles one pixel
    int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check — same reason as histogram kernels
    if (pixel_index >= width * height) return;

    // Read original RGB values for this pixel
    int r = d_image[pixel_index * 3 + 0];
    int g = d_image[pixel_index * 3 + 1];
    int b = d_image[pixel_index * 3 + 2];

    // Find the minimum non-zero CDF value
    // This is the CDF value of the darkest color that actually exists
    // in the image — needed for the normalization formula
    // For simplicity we use CDF[0] as approximation here
    // In a full implementation this would be precomputed separately
    unsigned int cdf_min_r = d_hist[0];
    unsigned int cdf_min_g = d_hist[256];
    unsigned int cdf_min_b = d_hist[512];

    // Apply histogram equalization formula to each channel
    // new = round((CDF[old] - CDF_min) / (total_pixels - CDF_min) * 255)
    // We clamp to 0-255 to ensure valid pixel values
    
    // Red channel — CDF values are in slots 0-255
    float new_r = (float)(d_hist[r] - cdf_min_r) /
                  (float)(total_pixels - cdf_min_r) * 255.0f;

    // Green channel — CDF values are in slots 256-511
    float new_g = (float)(d_hist[256 + g] - cdf_min_g) /
                  (float)(total_pixels - cdf_min_g) * 255.0f;

    // Blue channel — CDF values are in slots 512-767
    float new_b = (float)(d_hist[512 + b] - cdf_min_b) /
                  (float)(total_pixels - cdf_min_b) * 255.0f;

    // Write remapped values back to image
    // (unsigned char) cast clamps naturally to 0-255
    d_image[pixel_index * 3 + 0] = (unsigned char)min(255.0f, max(0.0f, new_r));
    d_image[pixel_index * 3 + 1] = (unsigned char)min(255.0f, max(0.0f, new_g));
    d_image[pixel_index * 3 + 2] = (unsigned char)min(255.0f, max(0.0f, new_b));
}