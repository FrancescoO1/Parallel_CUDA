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
#include "BenchmarkExporter.h"

namespace fs = std::filesystem;

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
    double throughput = (total_pixels / 1000000.0) / (avg / 1000.0);

    return BenchmarkStats{avg, std_dev, throughput};
}

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



// MAIN
int main() {
    const std::string imageDir = "/media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/4k_IMG/Dataset4K";
    const int NUM_ITERATIONS = 10;
    const int MAX_IMAGES_TOTAL = 20;

    std::vector<int> workloads = {1, 5, 10, 15, 20};
    BenchmarkExporter exporter;

    std::vector<std::string> image_files = getImageFiles(imageDir, MAX_IMAGES_TOTAL);
    if (image_files.empty()) {
        std::cerr << "Nessuna immagine trovata nella directory specificata!" << std::endl;
        return 1;
    }

    std::cout << "\n========== PRE-CARICAMENTO IMMAGINI ==========" << std::endl;
    std::cout << "Caricamento di " << image_files.size() << " immagini totali..." << std::endl;

    std::vector<Image> preloaded_images;
    for (const auto& path : image_files) {
        preloaded_images.emplace_back(path);
    }
    std::cout << "Caricamento completato." << std::endl;

    for (int num_images_to_process : workloads) {
        if (num_images_to_process > preloaded_images.size()) {
            std::cout << "\nSaltato workload " << num_images_to_process << " immagini (troppe poche immagini caricate)." << std::endl;
            continue;
        }

        std::cout << "\n\n========== INIZIO BENCHMARK PER " << num_images_to_process << " IMMAGINI ==========" << std::endl;

        size_t total_pixels_in_run = 0;
        std::vector<std::vector<float>> precomputed_grayscale;
        std::vector<size_t> precomputed_widths, precomputed_heights, precomputed_offsets;
        size_t batch_total_pixels = 0;

        for (int i = 0; i < num_images_to_process; ++i) {
            const auto& img = preloaded_images[i];
            precomputed_grayscale.push_back(img.toGrayscaleFloat());
            precomputed_widths.push_back(img.getWidth());
            precomputed_heights.push_back(img.getHeight());
            precomputed_offsets.push_back(batch_total_pixels);
            batch_total_pixels += img.getWidth() * img.getHeight();
        }
        total_pixels_in_run = batch_total_pixels;

        std::vector<float> mega_buffer(batch_total_pixels);
        size_t current_offset = 0;
        for (const auto& gray_img : precomputed_grayscale) {
            std::copy(gray_img.begin(), gray_img.end(), mega_buffer.begin() + current_offset);
            current_offset += gray_img.size();
        }
        std::cout << "Dati per " << num_images_to_process << " immagini pronti. Pixel totali: "
                  << total_pixels_in_run << " (" << total_pixels_in_run / 1000000.0 << " MP)" << std::endl;


        // --- BENCHMARK CPU ---
        std::cout << "\n=== CPU (Sequenziale) - " << NUM_ITERATIONS << " iterazioni ===" << std::endl;
        std::vector<double> cpu_times;
        ImageProcessingManager cpuManager;
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < precomputed_grayscale.size(); ++i) {
                std::vector<float> grayscale_copy = precomputed_grayscale[i];
                cpuManager.getProcessor().applySharpenFilter(grayscale_copy,
                                                             precomputed_widths[i], precomputed_heights[i]);
            }
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            cpu_times.push_back(time_ms);
        }
        BenchmarkStats cpuStats = calculateStats(cpu_times, total_pixels_in_run);
        printf("CPU completato. Tempo medio: %.2f ms\n", cpuStats.avg_time_ms);


        // --- BENCHMARK CUDA GPU ---
        std::cout << "\n=== CUDA - " << NUM_ITERATIONS << " iterazioni ===" << std::endl;
        std::vector<double> cuda_times;
        CudaImageProcessingManager cudaManager;

        float* h_mega_input_pinned = nullptr;
        float* h_mega_output_pinned = nullptr;
        cudaMallocHost(&h_mega_input_pinned, batch_total_pixels * sizeof(float));
        cudaMallocHost(&h_mega_output_pinned, batch_total_pixels * sizeof(float));
        std::copy(mega_buffer.begin(), mega_buffer.end(), h_mega_input_pinned);

        float* d_mega_input = nullptr;
        float* d_mega_output = nullptr;
        size_t* d_widths = nullptr;
        size_t* d_heights = nullptr;
        size_t* d_offsets = nullptr;
        cudaMalloc(&d_mega_input, batch_total_pixels * sizeof(float));
        cudaMalloc(&d_mega_output, batch_total_pixels * sizeof(float));
        cudaMalloc(&d_widths, precomputed_grayscale.size() * sizeof(size_t));
        cudaMalloc(&d_heights, precomputed_grayscale.size() * sizeof(size_t));
        cudaMalloc(&d_offsets, precomputed_grayscale.size() * sizeof(size_t));

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMemcpy(d_widths, precomputed_widths.data(), precomputed_grayscale.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_heights, precomputed_heights.data(), precomputed_grayscale.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, precomputed_offsets.data(), precomputed_grayscale.size() * sizeof(size_t), cudaMemcpyHostToDevice);

        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            cudaMemcpyAsync(d_mega_input, h_mega_input_pinned, batch_total_pixels * sizeof(float),
                            cudaMemcpyHostToDevice, stream);

            int threads_per_block = 256;
            int max_blocks = 1536;
            int blocks = std::min(max_blocks, (int)((batch_total_pixels + threads_per_block - 1) / threads_per_block));

            cudaManager.getProcessor().applySharpenFilterMegaBatchPreallocated(
                d_mega_input, d_mega_output, d_widths, d_heights, d_offsets,
                precomputed_grayscale.size(), blocks, threads_per_block);

            cudaStreamSynchronize(stream);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            cuda_times.push_back(time_ms);
        }

        BenchmarkStats cudaStats = calculateStats(cuda_times, total_pixels_in_run);
        printf("CUDA completato. Tempo medio: %.2f ms\n", cudaStats.avg_time_ms);

        if (num_images_to_process == workloads.back()) {
            std::cout << "\n Salvataggio immagini filtrate GPU (solo per ultimo workload)..." << std::endl;
            cudaMemcpy(h_mega_output_pinned, d_mega_output, batch_total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < precomputed_grayscale.size(); ++i) {
                std::vector<float> filtered(precomputed_grayscale[i].size());
                std::copy(h_mega_output_pinned + precomputed_offsets[i],
                          h_mega_output_pinned + precomputed_offsets[i] + filtered.size(),
                          filtered.begin());
                normalizeBuffer(filtered);
                std::string out_path = "output_gpu_" + std::to_string(i) + ".png";
                saveGrayscaleImage(filtered, precomputed_widths[i], precomputed_heights[i], out_path);
            }
            std::cout << "Immagini GPU salvate!" << std::endl;
        }

        cudaStreamDestroy(stream);
        cudaFree(d_mega_input);
        cudaFree(d_mega_output);
        cudaFree(d_widths);
        cudaFree(d_heights);
        cudaFree(d_offsets);
        cudaFreeHost(h_mega_input_pinned);
        cudaFreeHost(h_mega_output_pinned);

        exporter.addRun(num_images_to_process, total_pixels_in_run / 1000000.0, cpuStats, cudaStats);


    }

    // STAMPA
    exporter.printConsoleTable();
    exporter.exportToCSV("/media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/benchmark_results.csv");

    return 0;
}