// Header file — declares all histogram functions
// so main.cu can call them without seeing their implementation
// Version 1 here — naive, intentionally wrong, demonstrates race conditions
#ifndef HISTOGRAM_CUH  
// "include guard" — prevents this header from being included twice if multiple files include it

#define HISTOGRAM_CUH

// Number of possible values per channel (for each RGB 0-255)
#define NUM_BINS 256

// Total bins for RGB — three channels of 256 each
#define TOTAL_BINS 768

// the compiler does a find-and-replace — every place it sees NUM_BINS in the code it replaces it with 256. Every place it sees TOTAL_BINS it replaces with 768.


// declaration — just announcing it exists, defination in .cu file
__global__ void histogram_naive(   // this double underscore keyword is CUDA specific. It means "this function runs on the GPU but is called from the CPU.
    unsigned char* d_image,        // a pointer to the image data sitting in GPU memory. Unsigned char because each pixel channel value is 0-255, which fits exactly in one unsigned byte. The d_ prefix is a naming convention — d stands for device meaning GPU. Not required by CUDA but universally used so you always know which pointers point to GPU memory vs CPU memory.
    unsigned int* d_hist,          // pointer to the histogram array in GPU memory. Unsigned int because histogram bin counts can get large — a 4K image has 8 million pixels, so a single bin could hold counts up to 8 million, which fits in unsigned int but not in unsigned char.
    int width,                     // the image dimensions. The kernel needs these to know the total number of pixels and to stay within bounds.
    int height
);

#endif

// Declaration of Version 2 kernel — atomic operations
// Correct result but slower due to global memory contention
__global__ void histogram_atomic(
    unsigned char* d_image,
    unsigned int* d_hist,
    int width,
    int height
);


// Declaration of Version 3 kernel — privatized shared memory histogram
// Correct and fast — contention reduced to one merge per block
__global__ void histogram_privatized(
    unsigned char* d_image,
    unsigned int* d_hist,
    int width,
    int height
);