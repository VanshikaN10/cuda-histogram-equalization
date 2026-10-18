// main.cu
// Entry point — orchestrates the entire pipeline
// Loads image, runs all three histogram versions with benchmarking,
// runs equalization, saves output image, prints results

#include <iostream>
#include <opencv2/opencv.hpp>
#include "histogram.cuh"
#include "equalization.cuh"
#include "benchmark.cuh"

int main() {

    // ─── 1. LOAD IMAGE ───────────────────────────────────────────
    // Load image from disk using OpenCV
    // IMREAD_COLOR forces RGB even if image is grayscale
    cv::Mat image = cv::imread("../images/input.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: could not load image" << std::endl;
        return -1;
    }

    // Convert from BGR (OpenCV default) to RGB
    

    int width        = image.cols;
    int height       = image.rows;
    int total_pixels = width * height;
    int image_bytes  = total_pixels * 3; // 3 bytes per pixel (R, G, B)

    std::cout << "Image loaded: " << width << "x" << height
              << " (" << total_pixels << " pixels)" << std::endl;

    // ─── 2. ALLOCATE MEMORY ──────────────────────────────────────
    // CPU side — raw pointer to image data
    unsigned char* h_image = image.data;

    // GPU side — allocate memory for image and histogram
    unsigned char* d_image;
    unsigned int*  d_hist;

    cudaMalloc(&d_image, image_bytes);
    cudaMalloc(&d_hist,  768 * sizeof(unsigned int));

    // ─── 3. COPY IMAGE TO GPU ────────────────────────────────────
    // cudaMemcpy(destination, source, size, direction)
    cudaMemcpy(d_image, h_image, image_bytes, cudaMemcpyHostToDevice);

    // ─── 4. KERNEL LAUNCH CONFIGURATION ─────────────────────────
    // 256 threads per block — standard choice, multiple of warp size 32
    int threads_per_block = 256;
    // How many blocks needed to cover all pixels
    // Ceiling division — ensures we cover every pixel
    int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    // ─── 5. CPU BASELINE ─────────────────────────────────────────
    // Measure CPU histogram time using chrono for comparison
    unsigned int cpu_hist[768] = {0};

    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < total_pixels; i++) {
        cpu_hist[h_image[i * 3 + 0]]++;
        cpu_hist[256 + h_image[i * 3 + 1]]++;
        cpu_hist[512 + h_image[i * 3 + 2]]++;
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>
                     (cpu_end - cpu_start).count();

    // ─── 6. VERSION 1 — NAIVE (intentionally wrong) ──────────────
    // Reset histogram to zero before each version
    cudaMemset(d_hist, 0, 768 * sizeof(unsigned int));

    CudaTimer timer;
    timer_create(timer);

    // Warm up
    histogram_naive<<<blocks, threads_per_block>>>(d_image, d_hist, width, height);
    cudaDeviceSynchronize();

    // Timed run
    cudaMemset(d_hist, 0, 768 * sizeof(unsigned int));
    timer_start(timer);
    histogram_naive<<<blocks, threads_per_block>>>(d_image, d_hist, width, height);
    timer_stop(timer);
    float naive_time = timer_elapsed(timer);

    std::cout << "\n--- BENCHMARK RESULTS ---" << std::endl;
    std::cout << "CPU baseline:         " << cpu_time    << " ms" << std::endl;
    std::cout << "GPU naive (wrong):    " << naive_time  << " ms (incorrect result)" << std::endl;

    // ─── 7. VERSION 2 — ATOMIC ───────────────────────────────────
    cudaMemset(d_hist, 0, 768 * sizeof(unsigned int));

    // Warm up
    histogram_atomic<<<blocks, threads_per_block>>>(d_image, d_hist, width, height);
    cudaDeviceSynchronize();

    cudaMemset(d_hist, 0, 768 * sizeof(unsigned int));
    timer_start(timer);
    histogram_atomic<<<blocks, threads_per_block>>>(d_image, d_hist, width, height);
    timer_stop(timer);
    float atomic_time = timer_elapsed(timer);

    std::cout << "GPU atomic:           " << atomic_time << " ms" << std::endl;

    // ─── 8. VERSION 3 — PRIVATIZED ───────────────────────────────
    cudaMemset(d_hist, 0, 768 * sizeof(unsigned int));

    // Warm up
    histogram_privatized<<<blocks, threads_per_block>>>(d_image, d_hist, width, height);
    cudaDeviceSynchronize();

    cudaMemset(d_hist, 0, 768 * sizeof(unsigned int));
    timer_start(timer);
    histogram_privatized<<<blocks, threads_per_block>>>(d_image, d_hist, width, height);
    timer_stop(timer);
    float privatized_time = timer_elapsed(timer);

    std::cout << "GPU privatized:       " << privatized_time << " ms" << std::endl;
    std::cout << "Speedup (atomic):     " << cpu_time / atomic_time      << "x" << std::endl;
    std::cout << "Speedup (privatized): " << cpu_time / privatized_time  << "x" << std::endl;

    // ─── 9. EQUALIZATION ─────────────────────────────────────────
    // Use the privatized histogram result for equalization
    // d_hist now contains the final correct histogram
    // Run prefix sum to convert to CDF
    int shared_mem_size = 768 * sizeof(unsigned int);
    prefix_sum<<<1, 768, shared_mem_size>>>(d_hist, 768);
    cudaDeviceSynchronize();

    // Remap every pixel using the CDF
    remap_pixels<<<blocks, threads_per_block>>>(
        d_image, d_hist, width, height, total_pixels
    );
    cudaDeviceSynchronize();

    // ─── 10. COPY RESULT BACK TO CPU ────────────────────────────
    cudaMemcpy(h_image, d_image, image_bytes, cudaMemcpyDeviceToHost);

    // ─── 11. SAVE OUTPUT IMAGE ──────────────────────────────────
    // Convert back from RGB to BGR for OpenCV to save correctly
   
    cv::imwrite("../images/output.jpg", image);
    std::cout << "\nOutput saved to images/output.jpg" << std::endl;

    // ─── 12. SAVE BENCHMARK REPORT ──────────────────────────────
    FILE* report = fopen("../results/benchmark_report.txt", "w");
    fprintf(report, "=== CUDA Histogram Equalization Benchmark ===\n\n");
    fprintf(report, "Image size: %dx%d (%d pixels)\n\n", width, height, total_pixels);
    fprintf(report, "CPU baseline:         %.3f ms\n", cpu_time);
    fprintf(report, "GPU naive (wrong):    %.3f ms\n", naive_time);
    fprintf(report, "GPU atomic:           %.3f ms\n", atomic_time);
    fprintf(report, "GPU privatized:       %.3f ms\n", privatized_time);
    fprintf(report, "\nSpeedup (atomic):     %.2fx\n", cpu_time / atomic_time);
    fprintf(report, "Speedup (privatized): %.2fx\n",   cpu_time / privatized_time);
    fclose(report);

    // ─── 13. CLEANUP ────────────────────────────────────────────
    timer_destroy(timer);
    cudaFree(d_image);
    cudaFree(d_hist);

    return 0;
}