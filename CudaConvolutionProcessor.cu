#include "CudaConvolutionProcessor.h"
#include <iostream>
#include <stdexcept>

// =====================================================
// DEFINIZIONI GLOBALI E KERNEL
// =====================================================

// Il kernel viene copiato qui una sola volta all'avvio
__constant__ float d_sharpen_kernel_const[9];

// Definizione del membro statico della classe
constexpr float CudaConvolutionProcessor::sharpen_kernel[9];

// Macro per la gestione degli errori CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

// Kernel CUDA "Mega-Batch" (Grid-Stride Loop) cuore del calcolo
__global__ void convolutionMegaBatchKernel(const float* input, float* output,
                                            const size_t* widths, const size_t* heights,
                                            const size_t* offsets, int num_images) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Calcola il numero totale di pixel
    size_t total_pixels = offsets[num_images - 1] + widths[num_images - 1] * heights[num_images - 1];

    // Ogni thread processa più pixel con stride per massimizzare l'utilizzo
    for (size_t pixel_id = tid; pixel_id < total_pixels; pixel_id += total_threads) {

        // Trova l'immagine a cui appartiene questo pixel
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

        // Gestione bordi
        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
            output[pixel_id] = input[pixel_id];
            continue;
        }

        // Applica la convoluzione usando MEMORIA COSTANTE
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

// =====================================================
// IMPLEMENTAZIONE METODI DELLA CLASSE
// =====================================================

// Il costruttore si occupa solo di copiare il kernel
// in memoria costante.
CudaConvolutionProcessor::CudaConvolutionProcessor() {
    // Copia il kernel sharpen nella memoria costante (una sola volta!)
    CUDA_CHECK(cudaMemcpyToSymbol(d_sharpen_kernel_const, sharpen_kernel,
                                  9 * sizeof(float)));
}

// Il distruttore è vuoto, dato che tutta la gestione
// della memoria GPU è fatta esternamente nel main.
CudaConvolutionProcessor::~CudaConvolutionProcessor() {
}

void CudaConvolutionProcessor::applySharpenFilterMegaBatchPreallocated(
    float* d_input, float* d_output,
    size_t* d_widths, size_t* d_heights, size_t* d_offsets,
    int num_images, int blocks, int threads_per_block) {

    // Lancia il kernel
    convolutionMegaBatchKernel<<<blocks, threads_per_block>>>(
        d_input, d_output,
        d_widths, d_heights, d_offsets, num_images);

    // Sincronizza per assicurarsi che il kernel sia finito
    // prima che il main continui a misurare il tempo.
    CUDA_CHECK(cudaDeviceSynchronize());
}