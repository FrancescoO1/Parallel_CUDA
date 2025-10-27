#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include "ImageProcessingManager.h"
#include "CudaImageProcessingManager.h"
#include "BenchmarkStats.h"
#include <filesystem>
#include <algorithm>
#include "Image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>

namespace fs = std::filesystem;

// Funzione per ottenere i file immagine da una directory
std::vector<std::string> getImageFiles(const std::string& directory, int max_images = 20) {
    std::vector<std::string> image_files;
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tga"};
    try {
        if (!fs::exists(directory) || !fs::is_directory(directory)) {
            std::cerr << "Directory non trovata: " << directory << std::endl;
            return image_files;
        }
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    image_files.push_back(path);
                }
            }
        }
        std::sort(image_files.begin(), image_files.end());
        if (max_images > 0 && image_files.size() > static_cast<size_t>(max_images)) {
            image_files.resize(max_images);
        }
    } catch (const fs::filesystem_error& ex) {
        std::cerr << "Errore accesso filesystem: " << ex.what() << std::endl;
    }
    return image_files;
}

// Funzione per calcolare statistiche da un vettore di tempi
BenchmarkStats calculateStats(const std::vector<double>& times_ms, size_t total_pixels) {
    if (times_ms.empty()) {
        return {0.0, 0.0, 0.0};
    }
    double sum = 0.0;
    for (double t : times_ms) sum += t;
    double avg = sum / times_ms.size();

    double variance = 0.0;
    for (double t : times_ms) {
        variance += (t - avg) * (t - avg);
    }
    double std_dev = sqrt(variance / times_ms.size());

    // Throughput: Megapixel/secondo
    double throughput = (total_pixels / 1000000.0) / (avg / 1000.0);

    return BenchmarkStats{avg, std_dev, throughput};
}

// Funzione di stampa tabellare
void printComparison(const BenchmarkStats& cpu, const BenchmarkStats& cuda) {
    std::cout << "\n================== CONFRONTO FINALE ==================" << std::endl;
    std::cout << "| Modalità   | Tempo medio (ms) | Dev. Std (ms) | Throughput (MP/s) |" << std::endl;
    std::cout << "|------------|------------------|---------------|-------------------|" << std::endl;
    printf("| CPU        | %-16.2f | %-13.2f | %-17.2f |\n", cpu.avg_time_ms, cpu.stddev_time_ms, cpu.avg_throughput_mps);
    printf("| CUDA       | %-16.2f | %-13.2f | %-17.2f |\n", cuda.avg_time_ms, cuda.stddev_time_ms, cuda.avg_throughput_mps);
    std::cout << "======================================================" << std::endl;
    if (cuda.avg_time_ms > 0) {
        double speedup = cpu.avg_time_ms / cuda.avg_time_ms;
        std::cout << "\nSpeedup (CPU/CUDA): " << speedup << "x" << std::endl;
    }
}

// Funzione per salvare un'immagine grayscale float [0,1] come PNG
void saveGrayscaleImage(const std::vector<float>& buffer, int width, int height, const std::string& filename) {
    std::vector<unsigned char> img8u(width * height);
    for (size_t i = 0; i < buffer.size(); ++i) {
        float v = buffer[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        img8u[i] = static_cast<unsigned char>(v * 255.0f);
    }
    stbi_write_png(filename.c_str(), width, height, 1, img8u.data(), width);
}

// Funzione per normalizzare un buffer float in [0,1]
void normalizeBuffer(std::vector<float>& buffer) {
    if (buffer.empty()) return;
    float min_v = *std::min_element(buffer.begin(), buffer.end());
    float max_v = *std::max_element(buffer.begin(), buffer.end());
    if (max_v - min_v > 1e-6f) {
        for (auto& v : buffer) {
            v = (v - min_v) / (max_v - min_v);
        }
    } else {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
    }
}

int main() {
    const std::string imageDir = "/media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/4k_IMG/Dataset4K";
    const int NUM_ITERATIONS = 10;
    const int MAX_IMAGES = 20;

    std::vector<std::string> image_files = getImageFiles(imageDir, MAX_IMAGES);
    if (image_files.empty()) {
        std::cerr << "Nessuna immagine trovata nella directory specificata!" << std::endl;
        return 1;
    }

    std::cout << "\n========== BENCHMARK CPU vs CUDA ==========" << std::endl;
    std::cout << "Caricamento di " << image_files.size() << " immagini..." << std::endl;

    std::vector<Image> preloaded_images;
    size_t total_pixels = 0;
    for (const auto& path : image_files) {
        preloaded_images.emplace_back(path);
        total_pixels += preloaded_images.back().getWidth() * preloaded_images.back().getHeight();
    }
    std::cout << "Caricamento completato. Pixel totali per run: " << total_pixels
              << " (" << total_pixels / 1000000.0 << " MP)" << std::endl;

    std::cout << "\nPre-calcolo conversione grayscale..." << std::endl;
    std::vector<std::vector<float>> precomputed_grayscale;
    std::vector<size_t> precomputed_widths, precomputed_heights, precomputed_offsets;
    size_t batch_total_pixels = 0;

    for (const auto& img : preloaded_images) {
        precomputed_grayscale.push_back(img.toGrayscaleFloat());
        precomputed_widths.push_back(img.getWidth());
        precomputed_heights.push_back(img.getHeight());
        precomputed_offsets.push_back(batch_total_pixels);
        batch_total_pixels += img.getWidth() * img.getHeight();
    }

    std::vector<float> mega_buffer(batch_total_pixels);
    size_t current_offset = 0;
    for (const auto& gray_img : precomputed_grayscale) {
        std::copy(gray_img.begin(), gray_img.end(), mega_buffer.begin() + current_offset);
        current_offset += gray_img.size();
    }
    std::cout << "Pre-calcolo completato!" << std::endl;

    // --- BENCHMARK CPU (SEQUENZIALE) ---
    std::cout << "\n=== CPU (Sequenziale) - " << NUM_ITERATIONS << " iterazioni ===" << std::endl;
    std::vector<double> cpu_times;
    ImageProcessingManager cpuManager;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::cout << "CPU Iterazione " << (iter + 1) << "/" << NUM_ITERATIONS << "..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < precomputed_grayscale.size(); ++i) {
            std::vector<float> grayscale_copy = precomputed_grayscale[i]; // Copia per non modificare l'originale
            cpuManager.getProcessor().applySharpenFilter(grayscale_copy,
                                                         precomputed_widths[i], precomputed_heights[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        cpu_times.push_back(time_ms);
        std::cout << " " << time_ms << " ms" << std::endl;
    }
    BenchmarkStats cpuStats = calculateStats(cpu_times, total_pixels);

    // --- BENCHMARK CUDA ---
    std::cout << "\n=== CUDA - " << NUM_ITERATIONS << " iterazioni ===" << std::endl;
    std::vector<double> cuda_times;
    CudaImageProcessingManager cudaManager;

    // ===== 1. PRE-ALLOCAZIONE MEMORIA PINNED (HOST) E DEVICE (GPU) =====
    std::cout << "Pre-allocazione memoria Host (pinned) e Device (GPU)..." << std::endl;

    float* h_mega_input_pinned = nullptr;
    float* h_mega_output_pinned = nullptr;
    // *** FIX 1: Allocare la memoria pinned sull'host ***
    cudaMallocHost(&h_mega_input_pinned, batch_total_pixels * sizeof(float));
    cudaMallocHost(&h_mega_output_pinned, batch_total_pixels * sizeof(float));

    // Copia i dati dal buffer standard a quello pinned
    std::copy(mega_buffer.begin(), mega_buffer.end(), h_mega_input_pinned);

    float* d_mega_input = nullptr;
    float* d_mega_output = nullptr;
    size_t* d_widths = nullptr;
    size_t* d_heights = nullptr;
    size_t* d_offsets = nullptr;
    // *** FIX 2: Allocare tutta la memoria necessaria sulla GPU ***
    cudaMalloc(&d_mega_input, batch_total_pixels * sizeof(float));
    cudaMalloc(&d_mega_output, batch_total_pixels * sizeof(float));
    cudaMalloc(&d_widths, preloaded_images.size() * sizeof(size_t));
    cudaMalloc(&d_heights, preloaded_images.size() * sizeof(size_t));
    cudaMalloc(&d_offsets, preloaded_images.size() * sizeof(size_t));

    // ===== 2. CREAZIONE STREAM E TRASFERIMENTO METADATI (UNA SOLA VOLTA) =====
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // I metadati non cambiano, quindi li trasferiamo una sola volta fuori dal ciclo
    cudaMemcpy(d_widths, precomputed_widths.data(), preloaded_images.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heights, precomputed_heights.data(), preloaded_images.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, precomputed_offsets.data(), preloaded_images.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    std::cout << "Memoria pre-allocata e metadati trasferiti." << std::endl;

    // ===== 3. CICLO DI BENCHMARK =====
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::cout << "CUDA Iterazione " << (iter + 1) << "/" << NUM_ITERATIONS << "..." << std::flush;

        auto start = std::chrono::high_resolution_clock::now();

        // *** FIX 3: Usare trasferimenti ASINCRONI sullo stream ***
        cudaMemcpyAsync(d_mega_input, h_mega_input_pinned, batch_total_pixels * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // Configurazione del kernel
        int threads_per_block = 256;
        int max_blocks = 1536;
        int blocks = std::min(max_blocks, (int)((batch_total_pixels + threads_per_block - 1) / threads_per_block));

        // Lancia il kernel sullo stesso stream
        cudaManager.getProcessor().applySharpenFilterMegaBatchPreallocated(
            d_mega_input, d_mega_output, d_widths, d_heights, d_offsets,
            preloaded_images.size(), blocks, threads_per_block);

        // *** FIX 4: Sincronizzare lo stream per attendere il completamento di TUTTE le operazioni ***
        cudaStreamSynchronize(stream);

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        cuda_times.push_back(time_ms);

        std::cout << " " << time_ms << " ms" << std::endl;
    }

    // ===== 4. COPIA DEI RISULTATI E PULIZIA =====
    // Copia il risultato finale dalla GPU all'host per il salvataggio
    cudaMemcpy(h_mega_output_pinned, d_mega_output, batch_total_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    BenchmarkStats cudaStats = calculateStats(cuda_times, total_pixels);

    // --- SALVATAGGIO IMMAGINI ELABORATE GPU (FUORI DAL BENCHMARK) ---
    std::cout << "\n Salvataggio immagini filtrate GPU..." << std::endl;
    // Non è necessario rieseguire il kernel, i risultati sono già in h_mega_output_pinned
    for (size_t i = 0; i < preloaded_images.size(); ++i) {
        std::vector<float> filtered(precomputed_grayscale[i].size());
        // *** FIX 5: Copiare dal buffer di output pinnato (h_mega_output_pinned) ***
        std::copy(h_mega_output_pinned + precomputed_offsets[i],
                  h_mega_output_pinned + precomputed_offsets[i] + filtered.size(),
                  filtered.begin());
        normalizeBuffer(filtered);
        std::string out_path = "output_gpu_" + std::to_string(i) + ".png"; // Salva nella directory corrente
        saveGrayscaleImage(filtered, precomputed_widths[i], precomputed_heights[i], out_path);
    }
    std::cout << "Immagini GPU salvate!" << std::endl;

    // *** FIX 6: Liberare TUTTA la memoria allocata (stream, device e host) ***
    cudaStreamDestroy(stream);
    cudaFree(d_mega_input);
    cudaFree(d_mega_output);
    cudaFree(d_widths);
    cudaFree(d_heights);
    cudaFree(d_offsets);
    cudaFreeHost(h_mega_input_pinned);
    cudaFreeHost(h_mega_output_pinned);

    // --- CONFRONTO FINALE ---
    printComparison(cpuStats, cudaStats);

    return 0;
}