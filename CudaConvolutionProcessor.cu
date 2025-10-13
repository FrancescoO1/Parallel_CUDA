#include "CudaConvolutionProcessor.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>

// =====================================================
// OTTIMIZZAZIONE 1: MEMORIA COSTANTE per il kernel
// =====================================================
// La memoria costante è cached e broadcast a tutti i thread in un warp
// Ideale per il kernel di convoluzione che viene letto da tutti i thread
__constant__ float d_sharpen_kernel_const[9];

// Definizione della costante statica host
constexpr float CudaConvolutionProcessor::sharpen_kernel[9];

// =====================================================
// MACRO per GESTIONE ERRORI ROBUSTA (OTTIMIZAZIONE 2)
// =====================================================
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

// =====================================================
// KERNEL con SHARED MEMORY (TILING)
// =====================================================
// TILE_SIZE include il padding (1 pixel per lato per la convoluzione 3x3)
#define TILE_WIDTH 16
#define TILE_SIZE (TILE_WIDTH + 2)  // 18x18 per includere i bordi

__global__ void convolutionKernelOptimized(const float* input, float* output,
                                          int width, int height) {
    // Shared memory per il tile con padding
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    // Coordinate globali del thread
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    // Coordinate locali nel tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Carica il tile in shared memory con padding (cooperative loading)
    // Ogni thread carica il suo pixel + può caricare pixel di padding
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            int tile_x = tx * 2 + dx;
            int tile_y = ty * 2 + dy;

            if (tile_x < TILE_SIZE && tile_y < TILE_SIZE) {
                int global_x = blockIdx.x * TILE_WIDTH + tile_x - 1;
                int global_y = blockIdx.y * TILE_WIDTH + tile_y - 1;

                // Gestione bordi con clamping
                global_x = max(0, min(global_x, width - 1));
                global_y = max(0, min(global_y, height - 1));

                tile[tile_y][tile_x] = input[global_y * width + global_x];
            }
        }
    }

    // Sincronizza per assicurare che tutto il tile sia caricato
    __syncthreads();

    // Se il thread è fuori dai bounds, ritorna
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Gestione bordi: copia senza convoluzione
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[idx] = input[idx];
        return;
    }

    // Applica la convoluzione usando SHARED MEMORY (molto più veloce!)
    float sum = 0.0f;
    #pragma unroll
    for (int ky = 0; ky < 3; ky++) {
        #pragma unroll
        for (int kx = 0; kx < 3; kx++) {
            // Leggi dalla shared memory invece della global memory
            float pixel_val = tile[ty + ky][tx + kx];
            float kernel_val = d_sharpen_kernel_const[ky * 3 + kx];
            sum += pixel_val * kernel_val;
        }
    }

    // Clamp il risultato
    output[idx] = fmaxf(0.0f, fminf(255.0f, sum));
}

// =====================================================
// KERNEL  con SHARED MEMORY
// =====================================================
__global__ void convolutionMegaBatchKernelOptimized(const float* input, float* output,
                                                    const size_t* widths, const size_t* heights,
                                                    const size_t* offsets, int num_images) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Calcola il numero totale di pixel
    size_t total_pixels = offsets[num_images - 1] + widths[num_images - 1] * heights[num_images - 1];

    // Ogni thread processa più pixel con stride per massimizzare l'utilizzo
    for (size_t pixel_id = tid; pixel_id < total_pixels; pixel_id += total_threads) {

        // Trova l'immagine a cui appartiene questo pixel (binary search sarebbe meglio, ma per 20 immagini va bene linear)
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
// KERNEL BATCH OTTIMIZZATO (per immagini stessa dimensione)
// =====================================================
__global__ void convolutionKernelBatch(const float* input, float* output,
                                       int width, int height, int batch_size) {
    int img_idx = blockIdx.z; // Ogni blocco z lavora su una immagine diversa
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    if (x >= width || y >= height || img_idx >= batch_size) return;

    int img_offset = img_idx * width * height;
    int idx = img_offset + y * width + x;

    // Copia i pixel dei bordi senza applicare la convoluzione
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[idx] = input[idx];
        return;
    }

    float sum = 0.0f;
    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        #pragma unroll
        for (int kx = -1; kx <= 1; kx++) {
            int pixel_y = y + ky;
            int pixel_x = x + kx;

            if (pixel_y < 0 || pixel_y >= height || pixel_x < 0 || pixel_x >= width) continue;

            int pixel_idx = img_offset + pixel_y * width + pixel_x;
            int kernel_idx = (ky + 1) * 3 + (kx + 1);
            sum += input[pixel_idx] * d_sharpen_kernel_const[kernel_idx];
        }
    }
    output[idx] = fmaxf(0.0f, fminf(255.0f, sum));
}

CudaConvolutionProcessor::CudaConvolutionProcessor()
    : d_input(nullptr), d_output(nullptr), d_kernel(nullptr),
      max_width(0), max_height(0), initialized(false) {

    // Copia il kernel sharpen nella memoria costante (una sola volta!)
    CUDA_CHECK(cudaMemcpyToSymbol(d_sharpen_kernel_const, sharpen_kernel,
                                  9 * sizeof(float)));
}

CudaConvolutionProcessor::~CudaConvolutionProcessor() {
    cleanupGPUMemory();
}

void CudaConvolutionProcessor::initializeGPUMemory(size_t width, size_t height) {
    if (initialized && width <= max_width && height <= max_height) {
        return;
    }

    cleanupGPUMemory();

    size_t image_size = width * height * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, image_size));
    CUDA_CHECK(cudaMalloc(&d_output, image_size));

    max_width = width;
    max_height = height;
    initialized = true;
}

void CudaConvolutionProcessor::cleanupGPUMemory() {
    if (d_input) {
        cudaFree(d_input);
        d_input = nullptr;
    }
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
    if (d_kernel) {
        cudaFree(d_kernel);
        d_kernel = nullptr;
    }
    initialized = false;
}

std::vector<float> CudaConvolutionProcessor::applySharpenFilter(
    const std::vector<float>& image, size_t width, size_t height) {

    initializeGPUMemory(width, height);

    size_t image_size = width * height * sizeof(float);

    CUDA_CHECK(cudaMemcpy(d_input, image.data(), image_size, cudaMemcpyHostToDevice));

    // Usa TILE_WIDTH per la configurazione dei blocchi
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((width + TILE_WIDTH - 1) / TILE_WIDTH,
                  (height + TILE_WIDTH - 1) / TILE_WIDTH);

    // Usa il kernel OTTIMIZZATO con shared memory
    convolutionKernelOptimized<<<gridSize, blockSize>>>(d_input, d_output,
                                                        static_cast<int>(width),
                                                        static_cast<int>(height));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(width * height);
    CUDA_CHECK(cudaMemcpy(result.data(), d_output, image_size, cudaMemcpyDeviceToHost));

    return result;
}

std::vector<float> CudaConvolutionProcessor::applySharpenFilterTimed(
    const std::vector<float>& image, size_t width, size_t height,
    float& gpu_time_ms) {

    initializeGPUMemory(width, height);

    size_t image_size = width * height * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpy(d_input, image.data(), image_size, cudaMemcpyHostToDevice));

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((width + TILE_WIDTH - 1) / TILE_WIDTH,
                  (height + TILE_WIDTH - 1) / TILE_WIDTH);

    CUDA_CHECK(cudaEventRecord(start));

    convolutionKernelOptimized<<<gridSize, blockSize>>>(d_input, d_output,
                                                        static_cast<int>(width),
                                                        static_cast<int>(height));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    CUDA_CHECK(cudaGetLastError());

    std::vector<float> result(width * height);
    CUDA_CHECK(cudaMemcpy(result.data(), d_output, image_size, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

std::vector<std::vector<float>> CudaConvolutionProcessor::applySharpenFilterBatch(
    const std::vector<std::vector<float>>& images,
    size_t width, size_t height, float& gpu_time_ms) {
    size_t batch_size = images.size();
    if (batch_size == 0) {
        std::cerr << "[CUDA BATCH] Errore: batch_size == 0!" << std::endl;
        return {};
    }

    // Inizializza la memoria GPU per il kernel (necessario per d_kernel)
    initializeGPUMemory(width, height);

    size_t image_size = width * height;
    size_t total_size = batch_size * image_size;

    // Alloca buffer host unico
    std::vector<float> h_input(total_size);
    for (size_t i = 0; i < batch_size; ++i) {
        if (images[i].size() != image_size) {
            std::cerr << "[CUDA BATCH] Errore: immagine " << i << " ha dimensione " << images[i].size() << " invece di " << image_size << std::endl;
            return {};
        }
        std::copy(images[i].begin(), images[i].end(), h_input.begin() + i * image_size);
    }

    // Alloca memoria GPU
    float* d_input_batch = nullptr;
    float* d_output_batch = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_input_batch, total_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA BATCH] cudaMalloc d_input_batch failed: " << cudaGetErrorString(err) << std::endl;
        return {};
    }

    err = cudaMalloc(&d_output_batch, total_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA BATCH] cudaMalloc d_output_batch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_batch);
        return {};
    }

    // Copia input su GPU
    err = cudaMemcpy(d_input_batch, h_input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA BATCH] cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_batch);
        cudaFree(d_output_batch);
        return {};
    }

    // Configura i thread blocks - limita il numero massimo di blocchi Z
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                 (height + blockSize.y - 1) / blockSize.y,
                 std::min(static_cast<size_t>(65535), batch_size)); // Limite CUDA per gridSize.z

    // Eventi per timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Lancia kernel batch OTTIMIZZATO (non serve più passare d_kernel, usa la memoria costante)
    convolutionKernelBatch<<<gridSize, blockSize>>>(d_input_batch, d_output_batch,
                                                   static_cast<int>(width), static_cast<int>(height), static_cast<int>(batch_size));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA BATCH] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_batch);
        cudaFree(d_output_batch);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return {};
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Copia risultati su host
    std::vector<float> h_output(total_size);
    err = cudaMemcpy(h_output.data(), d_output_batch, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA BATCH] cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_batch);
        cudaFree(d_output_batch);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return {};
    }

    // Smonta il batch in immagini singole
    std::vector<std::vector<float>> results(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        results[i].assign(h_output.begin() + i * image_size, h_output.begin() + (i + 1) * image_size);
    }

    // Cleanup
    cudaFree(d_input_batch);
    cudaFree(d_output_batch);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return results;
}

std::vector<float> CudaConvolutionProcessor::applySharpenFilterMegaBatch(
    std::vector<float>& mega_buffer,
    const std::vector<size_t>& widths,
    const std::vector<size_t>& heights,
    const std::vector<size_t>& offsets,
    float& gpu_time_ms) {

    if (mega_buffer.empty() || widths.empty()) {
        return mega_buffer;
    }

    // Inizializza la memoria GPU per il kernel
    initializeGPUMemory(2048, 2048);

    size_t total_pixels = mega_buffer.size();
    size_t num_images = widths.size();

    std::cout << "[CUDA MEGA] Total pixels: " << total_pixels << ", Images: " << num_images << std::endl;

    // Alloca memoria GPU per il mega-batch
    float* d_mega_input = nullptr;
    float* d_mega_output = nullptr;
    size_t* d_widths = nullptr;
    size_t* d_heights = nullptr;
    size_t* d_offsets = nullptr;

    cudaError_t err;

    // Allocazioni
    err = cudaMalloc(&d_mega_input, total_pixels * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA MEGA] cudaMalloc d_mega_input failed: " << cudaGetErrorString(err) << std::endl;
        return mega_buffer;
    }

    err = cudaMalloc(&d_mega_output, total_pixels * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA MEGA] cudaMalloc d_mega_output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_mega_input);
        return mega_buffer;
    }

    err = cudaMalloc(&d_widths, num_images * sizeof(size_t));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA MEGA] cudaMalloc d_widths failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_mega_input);
        cudaFree(d_mega_output);
        return mega_buffer;
    }

    err = cudaMalloc(&d_heights, num_images * sizeof(size_t));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA MEGA] cudaMalloc d_heights failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_mega_input);
        cudaFree(d_mega_output);
        cudaFree(d_widths);
        return mega_buffer;
    }

    err = cudaMalloc(&d_offsets, num_images * sizeof(size_t));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA MEGA] cudaMalloc d_offsets failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_mega_input);
        cudaFree(d_mega_output);
        cudaFree(d_widths);
        cudaFree(d_heights);
        return mega_buffer;
    }

    // Trasferimenti H2D
    cudaMemcpy(d_mega_input, mega_buffer.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_widths, widths.data(), num_images * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heights, heights.data(), num_images * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), num_images * sizeof(size_t), cudaMemcpyHostToDevice);

    // Configurazione kernel ottimizzata per GTX 1660 Ti Mobile
    int threads_per_block = 128; // Ridotto per evitare "too many resources"
    int max_blocks = 1536; // Numero di CUDA cores della GTX 1660 Ti Mobile
    int blocks = std::min(max_blocks, (int)((total_pixels + threads_per_block - 1) / threads_per_block));
    blocks = std::max(blocks, 256); // Minimo per buona occupancy

    std::cout << "[CUDA MEGA] Configurazione GTX 1660 Ti: " << blocks << " blocchi, " << threads_per_block << " thread/blocco" << std::endl;

    // Eventi per timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Lancia il mega-kernel ottimizzato (usa memoria costante, non serve d_kernel)
    convolutionMegaBatchKernelOptimized<<<blocks, threads_per_block>>>(
        d_mega_input, d_mega_output,
        d_widths, d_heights, d_offsets, num_images);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Controlla errori kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA MEGA] Kernel failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "[CUDA MEGA] Kernel completato con successo in " << gpu_time_ms << " ms" << std::endl;
    }

    // Copia risultati D2H
    std::vector<float> result(total_pixels);
    cudaMemcpy(result.data(), d_mega_output, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_mega_input);
    cudaFree(d_mega_output);
    cudaFree(d_widths);
    cudaFree(d_heights);
    cudaFree(d_offsets);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

void CudaConvolutionProcessor::applySharpenFilterMegaBatchPreallocated(
    float* d_input, float* d_output,
    size_t* d_widths, size_t* d_heights, size_t* d_offsets,
    int num_images, int blocks, int threads_per_block) {

    // Usa il kernel OTTIMIZZATO con memoria costante
    convolutionMegaBatchKernelOptimized<<<blocks, threads_per_block>>>(
        d_input, d_output,
        d_widths, d_heights, d_offsets, num_images);

    CUDA_CHECK(cudaDeviceSynchronize());
}
