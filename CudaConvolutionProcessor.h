#ifndef CUDA_CONVOLUTION_PROCESSOR_H
#define CUDA_CONVOLUTION_PROCESSOR_H

#include <memory>

class CudaConvolutionProcessor {
private:
    static constexpr float sharpen_kernel[9] = {
        -2.0f, -2.0f, -2.0f,
        -2.0f, 17.0f, -2.0f,
        -2.0f, -2.0f, -2.0f
    };

public:
    CudaConvolutionProcessor();
    ~CudaConvolutionProcessor();

    CudaConvolutionProcessor(const CudaConvolutionProcessor&) = delete;
    CudaConvolutionProcessor& operator=(const CudaConvolutionProcessor&) = delete;

    void applySharpenFilterMegaBatchPreallocated(
        float* d_input, float* d_output,
        size_t* d_widths, size_t* d_heights, size_t* d_offsets,
        int num_images, int blocks, int threads_per_block);

};

#endif
