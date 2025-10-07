/*

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
#include <map>
#include "stb_image_write.h"

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
    std::cout << "| CPU        | " << cpu.avg_time_ms << "           | " << cpu.stddev_time_ms << "      | " << cpu.avg_throughput_mps << "         |" << std::endl;
    std::cout << "| CUDA       | " << cuda.avg_time_ms << "           | " << cuda.stddev_time_ms << "      | " << cuda.avg_throughput_mps << "         |" << std::endl;
    std::cout << "======================================================" << std::endl;
    double speedup = cpu.avg_time_ms / cuda.avg_time_ms;
    std::cout << "\nSpeedup (CPU/CUDA): " << speedup << "x" << std::endl;
}

// Funzione per salvare un'immagine grayscale float [0,1] come PNG QUESTA FUNZIONE DA ELIMINARE SE MODIFCA I BENCHMARK
void saveGrayscaleImage(const std::vector<float>& buffer, int width, int height, const std::string& filename) {
    std::vector<unsigned char> img8u(width * height);
    for (int i = 0; i < width * height; ++i) {
        float v = buffer[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        img8u[i] = static_cast<unsigned char>(v * 255.0f);
    }
    stbi_write_png(filename.c_str(), width, height, 1, img8u.data(), width);
}

// Funzione per normalizzare un buffer float in [0,1]
void normalizeBuffer(std::vector<float>& buffer) {
    float min_v = *std::min_element(buffer.begin(), buffer.end());
    float max_v = *std::max_element(buffer.begin(), buffer.end());
    if (max_v - min_v > 1e-6f) {
        for (auto& v : buffer) {
            v = (v - min_v) / (max_v - min_v);
        }
    } else {
        // Se tutti i valori sono uguali, portali a 0
        std::fill(buffer.begin(), buffer.end(), 0.0f);
    }
}

int main() {
    const std::string imageDir = "/media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/archive/sharp";
    const int NUM_ITERATIONS = 10;
    const int MAX_IMAGES = 20;

    // Carica direttamente 20 immagini senza filtro per dimensione
    std::vector<std::string> image_files = getImageFiles(imageDir, MAX_IMAGES);
    if (image_files.empty()) {
        std::cerr << "Nessuna immagine trovata!" << std::endl;
        return 1;
    }

    // Usa tutte le immagini disponibili (fino a 20)
    if (image_files.size() > MAX_IMAGES) {
        image_files.resize(MAX_IMAGES);
    }

    std::cout << "\nCaricamento di " << image_files.size() << " immagini..." << std::endl;

    // PRE-CARICA tutte le immagini in memoria per eliminare overhead I/O
    std::vector<Image> preloaded_images;
    size_t total_pixels = 0;

    for (const auto& path : image_files) {
        preloaded_images.emplace_back(path);
        total_pixels += preloaded_images.back().getWidth() * preloaded_images.back().getHeight();
        std::cout << "Caricata: " << path << " ("
                  << preloaded_images.back().getWidth() << "x"
                  << preloaded_images.back().getHeight() << ")" << std::endl;
    }

    std::cout << "\nPixel totali: " << total_pixels << " (" << total_pixels/1000000.0 << " MP)" << std::endl;

    // --- BENCHMARK CPU (SEQUENZIALE) ---
    std::cout << "\n=== CPU (Sequenziale) - " << NUM_ITERATIONS << " iterazioni ===" << std::endl;
    std::vector<double> cpu_times;

    ImageProcessingManager cpuManager;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::cout << "CPU Iterazione " << (iter + 1) << "/" << NUM_ITERATIONS << "..." << std::flush;

        auto start = std::chrono::high_resolution_clock::now();

        // Processa tutte le immagini PRE-CARICATE sequenzialmente
        for (const auto& image : preloaded_images) {
            std::vector<float> grayscale = image.toGrayscaleFloat();
            cpuManager.getProcessor().applySharpenFilter(grayscale, image.getWidth(), image.getHeight());
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        cpu_times.push_back(time_ms);

        std::cout << " " << time_ms << " ms" << std::endl;
    }

    BenchmarkStats cpuStats = calculateStats(cpu_times, total_pixels);

    // --- SALVATAGGIO IMMAGINI ELABORATE CPU ---
    std::cout << "\nSalvataggio immagini filtrate CPU in /media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/immagini_output_cpu ..." << std::endl;
    for (size_t i = 0; i < preloaded_images.size(); ++i) {
        std::vector<float> grayscale = preloaded_images[i].toGrayscaleFloat();
        // Applica il filtro e salva solo l'immagine filtrata
        std::vector<float> filtered = cpuManager.getProcessor().applySharpenFilter(grayscale, preloaded_images[i].getWidth(), preloaded_images[i].getHeight());
        normalizeBuffer(filtered);
        std::string out_path_filt = "/media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/immagini_output_cpu/output_cpu_" + std::to_string(i) + ".png";
        saveGrayscaleImage(filtered, preloaded_images[i].getWidth(), preloaded_images[i].getHeight(), out_path_filt);
    }
    std::cout << "Immagini salvate!" << std::endl;

    // --- BENCHMARK CUDA (ULTRA-OTTIMIZZATO ZERO-OVERHEAD) ---
    std::cout << "\n=== CUDA (Ultra-Ottimizzato Zero-Overhead) - " << NUM_ITERATIONS << " iterazioni ===" << std::endl;
    std::vector<double> cuda_times;

    CudaImageProcessingManager cudaManager;

    // PRE-INIZIALIZZA memoria GPU UNA SOLA VOLTA
    std::cout << "Pre-inizializzazione memoria GPU..." << std::endl;
    cudaManager.getProcessor().initializeGPUMemory(2048, 2048);

    // Pre-calcola tutti i dati una volta sola
    std::vector<std::vector<float>> precomputed_grayscale;
    std::vector<size_t> precomputed_widths, precomputed_heights, precomputed_offsets;
    std::vector<float> mega_buffer;
    size_t batch_total_pixels = 0;

    for (const auto& img : preloaded_images) {
        precomputed_grayscale.push_back(img.toGrayscaleFloat());
        precomputed_widths.push_back(img.getWidth());
        precomputed_heights.push_back(img.getHeight());
        precomputed_offsets.push_back(batch_total_pixels);
        batch_total_pixels += img.getWidth() * img.getHeight();
    }

    mega_buffer.resize(batch_total_pixels);
    size_t offset = 0;
    for (size_t i = 0; i < preloaded_images.size(); ++i) {
        std::copy(precomputed_grayscale[i].begin(), precomputed_grayscale[i].end(),
                  mega_buffer.begin() + offset);
        offset += precomputed_grayscale[i].size();
    }

    // Pre-alloca TUTTA la memoria GPU una volta sola
    float* d_mega_input = nullptr;
    float* d_mega_output = nullptr;
    size_t* d_widths = nullptr;
    size_t* d_heights = nullptr;
    size_t* d_offsets = nullptr;

    cudaMalloc(&d_mega_input, batch_total_pixels * sizeof(float));
    cudaMalloc(&d_mega_output, batch_total_pixels * sizeof(float));
    cudaMalloc(&d_widths, preloaded_images.size() * sizeof(size_t));
    cudaMalloc(&d_heights, preloaded_images.size() * sizeof(size_t));
    cudaMalloc(&d_offsets, preloaded_images.size() * sizeof(size_t));

    // Copia i metadati una volta sola (non cambiano mai)
    cudaMemcpy(d_widths, precomputed_widths.data(), preloaded_images.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heights, precomputed_heights.data(), preloaded_images.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, precomputed_offsets.data(), preloaded_images.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    std::cout << "Memoria GPU pre-allocata: " << batch_total_pixels << " pixels, " << preloaded_images.size() << " immagini" << std::endl;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::cout << "CUDA Iterazione " << (iter + 1) << "/" << NUM_ITERATIONS << "..." << std::flush;

        auto start = std::chrono::high_resolution_clock::now();

        // Ora ogni iterazione fa SOLO:
        // 1) Una copia H2D del mega-buffer
        // 2) Un kernel launch
        // 3) Una copia D2H del risultato (opzionale)

        cudaMemcpy(d_mega_input, mega_buffer.data(), mega_buffer.size() * sizeof(float), cudaMemcpyHostToDevice);

        // USA LA VERSIONE OTTIMIZZATA che riusa la memoria pre-allocata - ZERO allocazioni!
        cudaManager.getProcessor().applySharpenFilterMegaBatchPreallocated(
            d_mega_input, d_mega_output, d_widths, d_heights, d_offsets,
            preloaded_images.size(), 1536, 128);

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        cuda_times.push_back(time_ms);

        std::cout << " " << time_ms << " ms" << std::endl;
    }

    // PULISCE memoria GPU alla fine
    cudaFree(d_mega_input);
    cudaFree(d_mega_output);
    cudaFree(d_widths);
    cudaFree(d_heights);
    cudaFree(d_offsets);

    BenchmarkStats cudaStats = calculateStats(cuda_times, total_pixels);

    // --- CONFRONTO FINALE ---
    printComparison(cpuStats, cudaStats);

    return 0;
}
*/

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
#include <map>
#include "stb_image_write.h"

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
    double sum = 0.0;
    for (double t : times_ms) sum += t;
    double avg = sum / times_ms.size();

    double variance = 0.0;
    for (double t : times_ms) {
        variance += (t - avg) * (t - avg);
    }
    double std_dev = sqrt(variance / times_ms.size());

    double throughput = (total_pixels / 1000000.0) / (avg / 1000.0);

    return BenchmarkStats{avg, std_dev, throughput};
}

// Funzione di stampa tabellare
void printComparison(const BenchmarkStats& cpu, const BenchmarkStats& cuda) {
    std::cout << "\n================== CONFRONTO FINALE ==================" << std::endl;
    std::cout << "| Modalità   | Tempo medio (ms) | Dev. Std (ms) | Throughput (MP/s) |" << std::endl;
    std::cout << "|------------|------------------|---------------|-------------------|" << std::endl;
    std::cout << "| CPU        | " << cpu.avg_time_ms << "           | " << cpu.stddev_time_ms << "      | " << cpu.avg_throughput_mps << "         |" << std::endl;
    std::cout << "| CUDA       | " << cuda.avg_time_ms << "           | " << cuda.stddev_time_ms << "      | " << cuda.avg_throughput_mps << "         |" << std::endl;
    std::cout << "======================================================" << std::endl;
    double speedup = cpu.avg_time_ms / cuda.avg_time_ms;
    std::cout << "\nSpeedup (CPU/CUDA): " << speedup << "x" << std::endl;
}

// Funzione per salvare un'immagine grayscale float come PNG
void saveGrayscaleImage(const std::vector<float>& buffer, int width, int height, const std::string& filename) {
    std::vector<unsigned char> img8u(width * height);
    for (int i = 0; i < width * height; ++i) {
        float v = buffer[i];
        v = std::max(0.0f, std::min(255.0f, v));
        img8u[i] = static_cast<unsigned char>(v);
    }
    stbi_write_png(filename.c_str(), width, height, 1, img8u.data(), width);
}

int main() {
    const std::string imageDir = "/media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/archive/sharp";
    const int NUM_ITERATIONS = 10;
    const int MAX_IMAGES = 20;

    std::vector<std::string> image_files = getImageFiles(imageDir, MAX_IMAGES);
    if (image_files.empty()) {
        std::cerr << "Nessuna immagine trovata!" << std::endl;
        return 1;
    }

    if (image_files.size() > MAX_IMAGES) {
        image_files.resize(MAX_IMAGES);
    }

    std::cout << "\nCaricamento di " << image_files.size() << " immagini..." << std::endl;

    std::vector<Image> preloaded_images;
    size_t total_pixels = 0;

    for (const auto& path : image_files) {
        preloaded_images.emplace_back(path);
        total_pixels += preloaded_images.back().getWidth() * preloaded_images.back().getHeight();
        std::cout << "Caricata: " << path << " ("
                  << preloaded_images.back().getWidth() << "x"
                  << preloaded_images.back().getHeight() << ")" << std::endl;
    }

    std::cout << "\nPixel totali: " << total_pixels << " (" << total_pixels/1000000.0 << " MP)" << std::endl;

    // --- BENCHMARK CPU (SEQUENZIALE) ---
    std::cout << "\n=== CPU (Sequenziale) - " << NUM_ITERATIONS << " iterazioni ===" << std::endl;
    std::vector<double> cpu_times;

    ImageProcessingManager cpuManager;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::cout << "CPU Iterazione " << (iter + 1) << "/" << NUM_ITERATIONS << "..." << std::flush;

        auto start = std::chrono::high_resolution_clock::now();

        for (const auto& image : preloaded_images) {
            std::vector<float> grayscale = image.toGrayscaleFloat();
            cpuManager.getProcessor().applySharpenFilter(grayscale, image.getWidth(), image.getHeight());
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        cpu_times.push_back(time_ms);

        std::cout << " " << time_ms << " ms" << std::endl;
    }

    BenchmarkStats cpuStats = calculateStats(cpu_times, total_pixels);

    // --- BENCHMARK CUDA (ULTRA-OTTIMIZZATO ZERO-OVERHEAD) ---
    std::cout << "\n=== CUDA (Ultra-Ottimizzato Zero-Overhead) - " << NUM_ITERATIONS << " iterazioni ===" << std::endl;
    std::vector<double> cuda_times;

    CudaImageProcessingManager cudaManager;

    std::cout << "Pre-inizializzazione memoria GPU..." << std::endl;
    cudaManager.getProcessor().initializeGPUMemory(2048, 2048);

    std::vector<std::vector<float>> precomputed_grayscale;
    std::vector<size_t> precomputed_widths, precomputed_heights, precomputed_offsets;
    std::vector<float> mega_buffer;
    size_t batch_total_pixels = 0;

    for (const auto& img : preloaded_images) {
        precomputed_grayscale.push_back(img.toGrayscaleFloat());
        precomputed_widths.push_back(img.getWidth());
        precomputed_heights.push_back(img.getHeight());
        precomputed_offsets.push_back(batch_total_pixels);
        batch_total_pixels += img.getWidth() * img.getHeight();
    }

    mega_buffer.resize(batch_total_pixels);
    size_t offset = 0;
    for (size_t i = 0; i < preloaded_images.size(); ++i) {
        std::copy(precomputed_grayscale[i].begin(), precomputed_grayscale[i].end(),
                  mega_buffer.begin() + offset);
        offset += precomputed_grayscale[i].size();
    }

    float* d_mega_input = nullptr;
    float* d_mega_output = nullptr;
    size_t* d_widths = nullptr;
    size_t* d_heights = nullptr;
    size_t* d_offsets = nullptr;

    cudaMalloc(&d_mega_input, batch_total_pixels * sizeof(float));
    cudaMalloc(&d_mega_output, batch_total_pixels * sizeof(float));
    cudaMalloc(&d_widths, preloaded_images.size() * sizeof(size_t));
    cudaMalloc(&d_heights, preloaded_images.size() * sizeof(size_t));
    cudaMalloc(&d_offsets, preloaded_images.size() * sizeof(size_t));

    cudaMemcpy(d_widths, precomputed_widths.data(), preloaded_images.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heights, precomputed_heights.data(), preloaded_images.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, precomputed_offsets.data(), preloaded_images.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    std::cout << "Memoria GPU pre-allocata: " << batch_total_pixels << " pixels, " << preloaded_images.size() << " immagini" << std::endl;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::cout << "CUDA Iterazione " << (iter + 1) << "/" << NUM_ITERATIONS << "..." << std::flush;

        auto start = std::chrono::high_resolution_clock::now();

        cudaMemcpy(d_mega_input, mega_buffer.data(), mega_buffer.size() * sizeof(float), cudaMemcpyHostToDevice);

        cudaManager.getProcessor().applySharpenFilterMegaBatchPreallocated(
            d_mega_input, d_mega_output, d_widths, d_heights, d_offsets,
            preloaded_images.size(), 1536, 128);

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        cuda_times.push_back(time_ms);

        std::cout << " " << time_ms << " ms" << std::endl;
    }

    // --- SALVATAGGIO IMMAGINI ELABORATE GPU ---
    std::cout << "\nSalvataggio immagini filtrate GPU..." << std::endl;

    // Copia risultato finale dalla GPU
    std::vector<float> gpu_output(batch_total_pixels);
    cudaMemcpy(gpu_output.data(), d_mega_output, batch_total_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    // Salva ogni immagine separatamente
    for (size_t i = 0; i < preloaded_images.size(); ++i) {
        size_t img_offset = precomputed_offsets[i];
        size_t img_pixels = precomputed_widths[i] * precomputed_heights[i];

        std::vector<float> single_image(gpu_output.begin() + img_offset,
                                       gpu_output.begin() + img_offset + img_pixels);

        std::string out_path = "/media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/immagini_output_gpu/output_gpu_"
                              + std::to_string(i) + ".png";
        saveGrayscaleImage(single_image, precomputed_widths[i], precomputed_heights[i], out_path);
    }
    std::cout << "Immagini GPU salvate!" << std::endl;

    // Pulizia memoria GPU
    cudaFree(d_mega_input);
    cudaFree(d_mega_output);
    cudaFree(d_widths);
    cudaFree(d_heights);
    cudaFree(d_offsets);

    BenchmarkStats cudaStats = calculateStats(cuda_times, total_pixels);

    // --- CONFRONTO FINALE ---
    printComparison(cpuStats, cudaStats);

    return 0;
}
