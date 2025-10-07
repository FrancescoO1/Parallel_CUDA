#ifndef CUDA_CONVOLUTION_PROCESSOR_H
#define CUDA_CONVOLUTION_PROCESSOR_H

#include <cuda_runtime.h>
#include <vector>
#include <memory>

class CudaConvolutionProcessor {
private:
    float* d_input;
    float* d_output;
    float* d_kernel;
    size_t max_width;
    size_t max_height;
    bool initialized;

    // Kernel di sharpening 3x3 - ULTRA-AGGRESSIVO
    static constexpr float sharpen_kernel[9] = {
        -2.0f, -2.0f, -2.0f,
        -2.0f, 17.0f, -2.0f,
        -2.0f, -2.0f, -2.0f
    };

    void cleanupGPUMemory();

public:
    CudaConvolutionProcessor();
    ~CudaConvolutionProcessor();

    // Disabilita copy constructor e assignment operator
    CudaConvolutionProcessor(const CudaConvolutionProcessor&) = delete;
    CudaConvolutionProcessor& operator=(const CudaConvolutionProcessor&) = delete;

    std::vector<float> applySharpenFilter(const std::vector<float>& image, 
                                        size_t width, size_t height);
    
    // Nuovo metodo batch per processare più immagini contemporaneamente
    std::vector<std::vector<float>> applySharpenFilterBatch(
        const std::vector<std::vector<float>>& images,
        size_t width, size_t height, float& gpu_time_ms);

    // Mega-batch per processare tutte le immagini di dimensioni diverse in un singolo kernel
    std::vector<float> applySharpenFilterMegaBatch(
        std::vector<float>& mega_buffer,
        const std::vector<size_t>& widths,
        const std::vector<size_t>& heights,
        const std::vector<size_t>& offsets,
        float& gpu_time_ms);

    // Versione ottimizzata con memoria pre-allocata (zero overhead)
    void applySharpenFilterMegaBatchPreallocated(
        float* d_input, float* d_output,
        size_t* d_widths, size_t* d_heights, size_t* d_offsets,
        int num_images, int blocks, int threads_per_block);

    // RESO PUBBLICO per l'ottimizzazione zero-overhead
    void initializeGPUMemory(size_t width, size_t height);

    // Metodi per il benchmarking
    std::vector<float> applySharpenFilterTimed(const std::vector<float>& image, 
                                             size_t width, size_t height,
                                             float& gpu_time_ms);
};

#endif // CUDA_CONVOLUTION_PROCESSOR_H
