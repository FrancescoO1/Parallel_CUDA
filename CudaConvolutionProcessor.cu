#include "CudaConvolutionProcessor.h"
#include <iostream>
#include <stdexcept>

__constant__ float d_sharpen_kernel_const[9];

constexpr float CudaConvolutionProcessor::sharpen_kernel[9];

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

__global__ void convolutionMegaBatchKernel(const float* input, float* output,
                                            const size_t* widths, const size_t* heights,
                                            const size_t* offsets, int num_images) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    size_t total_pixels = offsets[num_images - 1] + widths[num_images - 1] * heights[num_images - 1];

    for (size_t pixel_id = tid; pixel_id < total_pixels; pixel_id += total_threads) {

        int img_idx = 0;
        for (int i = 0; i < num_images - 1; ++i) {
            if (pixel_id >= offsets[i + 1]) {
                img_idx = i + 1;
            } else {
                break;
            }
        }

        size_t width = widths[img_idx];
        size_t height = heights[img_idx];
        size_t offset = offsets[img_idx];

        int local_pixel = pixel_id - offset;
        int x = local_pixel % width;
        int y = local_pixel / width;

        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
            output[pixel_id] = input[pixel_id];
            continue;
        }

        float sum = 0.0f;
        #pragma unroll
        for (int ky = -1; ky <= 1; ky++) {
            #pragma unroll
            for (int kx = -1; kx <= 1; kx++) {
                int pixel_y = y + ky;
                int pixel_x = x + kx;

                int pixel_idx = offset + pixel_y * width + pixel_x;
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                sum += input[pixel_idx] * d_sharpen_kernel_const[kernel_idx];
            }
        }

        output[pixel_id] = fmaxf(0.0f, fminf(255.0f, sum));
    }
}

CudaConvolutionProcessor::CudaConvolutionProcessor() {
    CUDA_CHECK(cudaMemcpyToSymbol(d_sharpen_kernel_const, sharpen_kernel,
                                  9 * sizeof(float)));
}

CudaConvolutionProcessor::~CudaConvolutionProcessor() {
}

void CudaConvolutionProcessor::applySharpenFilterMegaBatchPreallocated(
    float* d_input, float* d_output,
    size_t* d_widths, size_t* d_heights, size_t* d_offsets,
    int num_images, int blocks, int threads_per_block) {

    convolutionMegaBatchKernel<<<blocks, threads_per_block>>>(
        d_input, d_output,
        d_widths, d_heights, d_offsets, num_images);

    CUDA_CHECK(cudaDeviceSynchronize());
}