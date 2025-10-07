#ifndef CUDA_IMAGE_PROCESSING_MANAGER_H
#define CUDA_IMAGE_PROCESSING_MANAGER_H

#include "Image.h"
#include "CudaConvolutionProcessor.h"
#include "PerformanceMonitor.h"
#include "BenchmarkStats.h"
#include <vector>
#include <string>

class CudaImageProcessingManager {
private:
    CudaConvolutionProcessor processor;
    PerformanceMonitor monitor;
    
    struct ProcessingResult {
        double cpu_time_ms;
        double gpu_time_ms;
        double total_time_ms;
        double throughput_mps;
        size_t total_pixels;
    };

public:
    void processImagesCUDA(const std::vector<std::string>& imagePaths);
    void processSingleImageCUDA(const std::string& imagePath);
    void processBatchImages(const std::vector<std::string>& imagePaths); // Nuova funzione per batch processing ottimizzato
    BenchmarkStats processImagesAndGetStats(const std::vector<std::string>& imageFiles);

    // Getter per accesso diretto al processor (per benchmark ottimizzato)
    CudaConvolutionProcessor& getProcessor() { return processor; }

    // Funzione per processare tutte le immagini in un mega-batch per massimo speedup
    void processMegaBatch(const std::vector<Image>& images);

    // Versione ottimizzata che pre-alloca memoria GPU
    void processMegaBatchOptimized(const std::vector<Image>& images);

private:
    ProcessingResult processImageBenchmark(const Image& image, int run_number);
    void printDetailedResults(const std::vector<ProcessingResult>& results, 
                            const std::string& test_name);
    double calculateSpeedup(double cpu_time, double gpu_time);
};

#endif // CUDA_IMAGE_PROCESSING_MANAGER_H
