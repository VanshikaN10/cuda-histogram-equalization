// Declares timing utilities for measuring CUDA kernel execution time. This whole #ifndef BENCHMARK_CUH #define BENCHMARK_CUH is called include gaurd. 
#ifndef BENCHMARK_CUH  // #ifndef means "if not defined yet." if the file BENCHMARK_CUH is not defined yet
#define BENCHMARK_CUH  // then define it. Prevents this header from being processed twice if multiple files include it.


#include <cuda_runtime.h>    // Pulls in CUDA's runtime library. cuda_runtime.h gives us cudaEvent_t and related functions


// just a container for holding a pair of two related CUDA events used to measure GPU kernel execution time 
struct CudaTimer {
    cudaEvent_t start;  // marker placed before kernel
    cudaEvent_t stop;   // marker placed after kernel
};


// Creates the timer — allocates the two events on GPU
void timer_create(CudaTimer& timer);


// Places the start marker on the GPU timeline
// Call this immediately before launching a kernel
void timer_start(CudaTimer& timer);


// Places the stop marker on the GPU timeline
// Call this immediately after launching a kernel
void timer_stop(CudaTimer& timer);


// Waits for GPU to finish and returns elapsed time in milliseconds
// Call this after timer_stop
float timer_elapsed(CudaTimer& timer);


// Cleans up — destroys the two events and frees resources
void timer_destroy(CudaTimer& timer);


// Runs a complete benchmark — warm up run + N timed runs + returns average
// Takes a function pointer so it can time any kernel launch
// num_runs = how many times to run for averaging
float benchmark_kernel(void (*kernel_launch)(), int num_runs = 10);

#endif