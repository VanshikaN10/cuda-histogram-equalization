//Declares timing utilities for measuring CUDA kernel execution time
#include "benchmark.cuh"
#include <stdio.h>


void timer_create(CudaTimer& timer) {
    // cudaEventCreate allocates a new event object on the GPU We need two — start,stop & passes the address so the function can fill in the event handle
    cudaEventCreate(&timer.start);
    cudaEventCreate(&timer.stop);
}


void timer_start(CudaTimer& timer) {
    // cudaEventRecord inserts the start event into the GPU's command queue at this exact position. The GPU processes its queue in order — when it reaches this marker it records its current timestamp.
    cudaEventRecord(timer.start);
}


void timer_stop(CudaTimer& timer) {
    //// cudaEventRecord inserts the stop event into the GPU's command queue at this exact position. The GPU processes its queue in order — when it reaches this marker it stops recording after this timestamp.
    cudaEventRecord(timer.stop);
}


float timer_elapsed(CudaTimer& timer) {
    // Remember the CPU doesn't wait for the GPU. After timer_stop places the stop marker in the queue the CPU immediately moves to this next line. But the GPU hasn't processed the stop marker yet — it's still in the queue, kernel might still be running.
    cudaEventSynchronize(timer.stop);

    // Ask GPU for the time difference between start and stop events Result is in milliseconds automatically
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, timer.start, timer.stop);
    
    return milliseconds;
}


void timer_destroy(CudaTimer& timer) {
    // Free the event objects — always clean up GPU resources
    // Same principle as deleting heap allocated memory in C++
    cudaEventDestroy(timer.start);
    cudaEventDestroy(timer.stop);
}


float benchmark_kernel(void (*kernel_launch)(), int num_runs) {  // kernel_launch is a pointer to a function that takes no arguments and returns void here written to trigger the kernel_launch function. 
    // Creates a CudaTimer and initializes its two events.
    CudaTimer timer;
    timer_create(timer);

    kernel_launch(); // launches the GPU kernel
    cudaDeviceSynchronize();  // this makes the CPU wait until that kernel has finished

    // Timed runs — run kernel num_runs times and accumulate time
    float total_time = 0;
    for (int i = 0; i < num_runs; i++) {
        timer_start(timer);
        kernel_launch();        
        timer_stop(timer);
        total_time += timer_elapsed(timer);  

    timer_destroy(timer);
    }

    // Return average time across all runs
    // More runs = more stable number = more credible benchmark
    return total_time / num_runs;
}