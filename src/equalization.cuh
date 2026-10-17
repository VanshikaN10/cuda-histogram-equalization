// equalization.cuh
// Declares the prefix sum kernel and pixel remap kernel
// These two kernels complete the equalization pipeline
// after the histogram has been computed

#ifndef EQUALIZATION_CUH
#define EQUALIZATION_CUH

#include <cuda_runtime.h>

// Prefix sum kernel
// Takes the 768-bin RGB histogram and computes
// cumulative distribution function (CDF) in place
// d_hist = histogram array on GPU, modified in place to become CDF
// num_bins = total number of bins (768 for RGB)
__global__ void prefix_sum(
    unsigned int* d_hist,
    int num_bins
);

// Pixel remap kernel
// Uses the CDF to remap every pixel to its equalized value
// d_image = original image on GPU, modified in place
// d_hist  = CDF array computed by prefix_sum kernel
// width   = image width
// height  = image height
// total_pixels = width * height, needed for normalization formula
__global__ void remap_pixels(
    unsigned char* d_image,
    unsigned int* d_hist,
    int width,
    int height,
    int total_pixels
);

#endif