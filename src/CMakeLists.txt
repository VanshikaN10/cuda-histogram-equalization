cmake_minimum_required(VERSION 3.18)
project(CudaHistogramEqualization CUDA CXX)

# CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# All source files
add_executable(histogram_eq
    src/main.cu
    src/histogram.cu
    src/equalization.cu
    src/benchmark.cu
)

# Link OpenCV
target_link_libraries(histogram_eq ${OpenCV_LIBS})

# GPU architecture — sm_75 is T4 (Colab's GPU)
set_target_properties(histogram_eq PROPERTIES
    CUDA_ARCHITECTURES "75"
)