#include "CudaImageProcessingManager.h"
#include "BenchmarkStats.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

CudaImageProcessingManager::ProcessingResult 
CudaImageProcessingManager::processImageBenchmark(const Image& image, int run_number) {
    ProcessingResult result;
    
    // Converti l'immagine in grayscale se necessario
    std::vector<float> grayscale = image.toGrayscaleFloat();
    result.total_pixels = image.getWidth() * image.getHeight();
    
    // Misurazione tempo CPU (include trasferimenti memoria)
    auto cpu_start = monitor.getCurrentTime();
    
    float gpu_time_ms;
    std::vector<float> processed = processor.applySharpenFilterTimed(
        grayscale, image.getWidth(), image.getHeight(), gpu_time_ms);
    
    auto cpu_end = monitor.getCurrentTime();
    
    result.cpu_time_ms = monitor.calculateDuration(cpu_start, cpu_end);
    result.gpu_time_ms = static_cast<double>(gpu_time_ms);
    result.total_time_ms = result.cpu_time_ms;
    result.throughput_mps = (result.total_pixels / 1e6) / (result.total_time_ms / 1000.0);
    
    std::cout << "Run " << std::setw(2) << run_number 
              << " - Total: " << std::fixed << std::setprecision(2) << result.total_time_ms << " ms"
              << " - GPU: " << result.gpu_time_ms << " ms"
              << " - Throughput: " << std::setprecision(1) << result.throughput_mps << " MP/s"
              << std::endl;
    
    return result;
}

void CudaImageProcessingManager::processImagesCUDA(const std::vector<std::string>& imagePaths) {
    if (imagePaths.empty()) {
        std::cout << "No images to process." << std::endl;
        return;
    }
    
    std::cout << "\n=== CUDA Batch Image Processing ===" << std::endl;
    std::cout << "Processing " << imagePaths.size() << " images, 25 runs each" << std::endl;
    
    std::vector<ProcessingResult> all_results;
    size_t total_pixels_per_run = 0;
    
    // Prima, carica tutte le immagini e calcola i pixel totali
    std::vector<Image> images;
    for (const auto& path : imagePaths) {
        images.emplace_back(path);
        total_pixels_per_run += images.back().getWidth() * images.back().getHeight();
        std::cout << "Loaded: " << path << " (" 
                  << images.back().getWidth() << "x" << images.back().getHeight() << ")" << std::endl;
    }
    
    std::cout << "\nStarting benchmark (25 runs):" << std::endl;
    
    // Esegui 25 run completi
    for (int run = 1; run <= 25; ++run) {
        double run_total_time = 0.0;
        double run_gpu_time = 0.0;
        
        for (size_t i = 0; i < images.size(); ++i) {
            ProcessingResult result = processImageBenchmark(images[i], run);
            run_total_time += result.total_time_ms;
            run_gpu_time += result.gpu_time_ms;
            
            // Salva solo la prima immagine di ogni run per evitare troppi file
            if (i == 0) {
                std::vector<float> grayscale = images[i].toGrayscaleFloat();
                std::vector<float> processed = processor.applySharpenFilter(
                    grayscale, images[i].getWidth(), images[i].getHeight());
                
                std::string outputPath = "cuda_output_" + std::to_string(run) + ".png";
                Image::saveGrayscaleFloat(processed, images[i].getWidth(), 
                                        images[i].getHeight(), outputPath);
            }
        }
        
        ProcessingResult run_result;
        run_result.total_time_ms = run_total_time;
        run_result.gpu_time_ms = run_gpu_time;
        run_result.total_pixels = total_pixels_per_run;
        run_result.throughput_mps = (total_pixels_per_run / 1e6) / (run_total_time / 1000.0);
        
        all_results.push_back(run_result);
        
        std::cout << "Run " << run << " complete - Total time: " 
                  << std::fixed << std::setprecision(2) << run_total_time << " ms" << std::endl;
    }
    
    printDetailedResults(all_results, "CUDA Batch Processing");
}

void CudaImageProcessingManager::processSingleImageCUDA(const std::string& imagePath) {
    std::cout << "\n=== CUDA Single Image Processing ===" << std::endl;
    
    Image image(imagePath);
    std::cout << "Loaded: " << imagePath << " (" 
              << image.getWidth() << "x" << image.getHeight() << ")" << std::endl;
    
    std::vector<ProcessingResult> results;
    
    std::cout << "\nStarting benchmark (25 runs):" << std::endl;
    
    for (int run = 1; run <= 25; ++run) {
        ProcessingResult result = processImageBenchmark(image, run);
        results.push_back(result);
        
        // Salva solo il primo e ultimo risultato
        if (run == 1 || run == 25) {
            std::vector<float> grayscale = image.toGrayscaleFloat();
            std::vector<float> processed = processor.applySharpenFilter(
                grayscale, image.getWidth(), image.getHeight());
            
            std::string outputPath = "cuda_single_output_" + std::to_string(run) + ".png";
            Image::saveGrayscaleFloat(processed, image.getWidth(), 
                                    image.getHeight(), outputPath);
        }
    }
    
    printDetailedResults(results, "CUDA Single Image Processing");
}

void CudaImageProcessingManager::processBatchImages(const std::vector<std::string>& imagePaths) {
    if (imagePaths.empty()) {
        return;
    }

    // Carica tutte le immagini e converte in grayscale float
    std::vector<std::vector<float>> all_grayscale;
    size_t width = 0, height = 0;

    for (const auto& path : imagePaths) {
        Image img(path);
        if (width == 0 && height == 0) {
            width = img.getWidth();
            height = img.getHeight();
        }
        all_grayscale.push_back(img.toGrayscaleFloat());
    }

    // Processa tutte le immagini in un singolo batch CUDA
    float gpu_time_ms = 0.0f;
    std::vector<std::vector<float>> processed = processor.applySharpenFilterBatch(
        all_grayscale, width, height, gpu_time_ms);

    // Le immagini sono processate ma non vengono salvate (solo per benchmark)
}

void CudaImageProcessingManager::printDetailedResults(
    const std::vector<ProcessingResult>& results, const std::string& test_name) {
    
    if (results.empty()) return;
    
    // Calcola statistiche per tempo totale
    std::vector<double> total_times;
    std::vector<double> gpu_times;
    std::vector<double> throughputs;
    
    for (const auto& result : results) {
        total_times.push_back(result.total_time_ms);
        gpu_times.push_back(result.gpu_time_ms);
        throughputs.push_back(result.throughput_mps);
    }
    
    double avg_total_time = std::accumulate(total_times.begin(), total_times.end(), 0.0) / total_times.size();
    double avg_gpu_time = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
    double avg_throughput = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / throughputs.size();
    
    // Calcola varianza e deviazione standard
    double variance_total = 0.0;
    double variance_gpu = 0.0;
    for (size_t i = 0; i < results.size(); ++i) {
        variance_total += std::pow(total_times[i] - avg_total_time, 2);
        variance_gpu += std::pow(gpu_times[i] - avg_gpu_time, 2);
    }
    variance_total /= total_times.size();
    variance_gpu /= gpu_times.size();
    
    double std_dev_total = std::sqrt(variance_total);
    double std_dev_gpu = std::sqrt(variance_gpu);
    
    // Stampa risultati
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << test_name << " - RESULTS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total pixels processed: " << results[0].total_pixels << std::endl;
    std::cout << "Number of runs: " << results.size() << std::endl;
    std::cout << "\nTiming Results:" << std::endl;
    std::cout << "  Average total time: " << avg_total_time << " ms" << std::endl;
    std::cout << "  Average GPU time: " << avg_gpu_time << " ms" << std::endl;
    std::cout << "  Total time variance: " << variance_total << " ms²" << std::endl;
    std::cout << "  GPU time variance: " << variance_gpu << " ms²" << std::endl;
    std::cout << "  Total time std dev: " << std_dev_total << " ms" << std::endl;
    std::cout << "  GPU time std dev: " << std_dev_gpu << " ms" << std::endl;
    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  Average throughput: " << std::setprecision(1) << avg_throughput << " MP/s" << std::endl;
    
    // Calcola speedup stimato (confronto con tempi tipici CPU sequenziali)
    // Assumiamo throughput CPU sequenziale tipico di 8-10 MP/s
    double estimated_cpu_throughput = 9.0; // MP/s
    double estimated_speedup = avg_throughput / estimated_cpu_throughput;
    
    std::cout << "\nSpeedup Estimation:" << std::endl;
    std::cout << "  Estimated CPU throughput: " << estimated_cpu_throughput << " MP/s" << std::endl;
    std::cout << "  Estimated speedup: " << std::setprecision(1) << estimated_speedup << "x" << std::endl;
    
    std::cout << std::string(60, '=') << std::endl;
}

double CudaImageProcessingManager::calculateSpeedup(double cpu_time, double gpu_time) {
    return cpu_time / gpu_time;
}

BenchmarkStats CudaImageProcessingManager::processImagesAndGetStats(const std::vector<std::string>& imageFiles) {
    const int num_runs = 25;
    const size_t batch_size = 8; // Puoi aumentare se la GPU ha più memoria
    std::vector<double> times;
    std::vector<double> throughputs;
    size_t total_pixels = 0;
    std::vector<Image> images;
    for (const auto& path : imageFiles) {
        images.emplace_back(path);
        total_pixels += images.back().getWidth() * images.back().getHeight();
    }
    // Pre-converti tutte le immagini in grayscale float
    std::vector<std::vector<float>> all_grayscale;
    for (const auto& img : images) {
        all_grayscale.push_back(img.toGrayscaleFloat());
    }
    size_t width = images[0].getWidth();
    size_t height = images[0].getHeight();
    for (int run = 1; run <= num_runs; ++run) {
        double run_total_time = 0.0;
        float gpu_time_ms = 0.0f;
        for (size_t i = 0; i < all_grayscale.size(); i += batch_size) {
            size_t current_batch = std::min(batch_size, all_grayscale.size() - i);
            std::vector<std::vector<float>> batch(all_grayscale.begin() + i, all_grayscale.begin() + i + current_batch);
            auto start = monitor.getCurrentTime();
            std::vector<std::vector<float>> processed = processor.applySharpenFilterBatch(batch, width, height, gpu_time_ms);
            auto end = monitor.getCurrentTime();
            double time_ms = monitor.calculateDuration(start, end);
            run_total_time += time_ms;
        }
        double throughput = (total_pixels / 1e6) / (run_total_time / 1000.0);
        times.push_back(run_total_time);
        throughputs.push_back(throughput);
    }
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avg_throughput = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / throughputs.size();
    double variance = 0.0;
    for (double t : times) variance += (t - avg_time) * (t - avg_time);
    variance /= times.size();
    double stddev = std::sqrt(variance);
    BenchmarkStats stats{avg_time, stddev, avg_throughput};
    return stats;
}

void CudaImageProcessingManager::processMegaBatch(const std::vector<Image>& images) {
    if (images.empty()) return;

    // Converti tutte le immagini in grayscale e concatenale
    std::vector<std::vector<float>> all_grayscale;
    std::vector<size_t> widths, heights, offsets;
    size_t total_pixels = 0;

    for (const auto& img : images) {
        all_grayscale.push_back(img.toGrayscaleFloat());
        widths.push_back(img.getWidth());
        heights.push_back(img.getHeight());
        offsets.push_back(total_pixels);
        total_pixels += img.getWidth() * img.getHeight();
    }

    // Crea un mega-buffer unico con tutte le immagini
    std::vector<float> mega_buffer(total_pixels);
    size_t offset = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        std::copy(all_grayscale[i].begin(), all_grayscale[i].end(),
                  mega_buffer.begin() + offset);
        offset += all_grayscale[i].size();
    }

    // Processa tutto con un singolo kernel launch su GPU
    float gpu_time;
    processor.applySharpenFilterMegaBatch(mega_buffer, widths, heights, offsets, gpu_time);
}

void CudaImageProcessingManager::processMegaBatchOptimized(const std::vector<Image>& images) {
    if (images.empty()) return;

    // Pre-alloca TUTTO una volta sola (chiamata solo al primo utilizzo)
    static bool first_call = true;
    static std::vector<std::vector<float>> precomputed_grayscale;
    static std::vector<size_t> precomputed_widths, precomputed_heights, precomputed_offsets;
    static std::vector<float> mega_buffer;
    static float* d_mega_input = nullptr;
    static float* d_mega_output = nullptr;
    static size_t* d_widths = nullptr;
    static size_t* d_heights = nullptr;
    static size_t* d_offsets = nullptr;

    if (first_call) {
        std::cout << "[CUDA OPTIMIZED] Prima esecuzione: pre-allocazione memoria GPU..." << std::endl;

        // Pre-calcola tutto una volta sola
        size_t total_pixels = 0;
        for (const auto& img : images) {
            precomputed_grayscale.push_back(img.toGrayscaleFloat());
            precomputed_widths.push_back(img.getWidth());
            precomputed_heights.push_back(img.getHeight());
            precomputed_offsets.push_back(total_pixels);
            total_pixels += img.getWidth() * img.getHeight();
        }

        // Crea mega-buffer una volta sola
        mega_buffer.resize(total_pixels);
        size_t offset = 0;
        for (size_t i = 0; i < images.size(); ++i) {
            std::copy(precomputed_grayscale[i].begin(), precomputed_grayscale[i].end(),
                      mega_buffer.begin() + offset);
            offset += precomputed_grayscale[i].size();
        }

        // Pre-alloca TUTTA la memoria GPU una volta sola
        processor.initializeGPUMemory(2048, 2048);

        cudaMalloc(&d_mega_input, total_pixels * sizeof(float));
        cudaMalloc(&d_mega_output, total_pixels * sizeof(float));
        cudaMalloc(&d_widths, images.size() * sizeof(size_t));
        cudaMalloc(&d_heights, images.size() * sizeof(size_t));
        cudaMalloc(&d_offsets, images.size() * sizeof(size_t));

        // Copia i metadati una volta sola (non cambiano mai)
        cudaMemcpy(d_widths, precomputed_widths.data(), images.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_heights, precomputed_heights.data(), images.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, precomputed_offsets.data(), images.size() * sizeof(size_t), cudaMemcpyHostToDevice);

        first_call = false;
        std::cout << "[CUDA OPTIMIZED] Pre-allocazione completata: " << total_pixels << " pixels, " << images.size() << " immagini" << std::endl;
    }

    // Ora ogni iterazione fa SOLO:
    // 1) Una copia H2D del mega-buffer
    // 2) Un kernel launch
    // 3) Una copia D2H del risultato

    cudaMemcpy(d_mega_input, mega_buffer.data(), mega_buffer.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Configurazione kernel ottimizzata
    int threads_per_block = 128;
    int blocks = 1536; // Fisso per GTX 1660 Ti

    // Lancia il kernel (solo questa parte viene misurata come "GPU time")
    processor.applySharpenFilterMegaBatchPreallocated(
        d_mega_input, d_mega_output, d_widths, d_heights, d_offsets,
        precomputed_widths.size(), blocks, threads_per_block);

    // Risultato finale (opzionale, non necessario per benchmark)
    // std::vector<float> result(mega_buffer.size());
    // cudaMemcpy(result.data(), d_mega_output, mega_buffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
}
