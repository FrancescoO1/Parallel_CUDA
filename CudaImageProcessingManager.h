#ifndef CUDA_IMAGE_PROCESSING_MANAGER_H
#define CUDA_IMAGE_PROCESSING_MANAGER_H

#include "CudaConvolutionProcessor.h"
#include <vector>
#include <string>

class CudaImageProcessingManager {
private:
    CudaConvolutionProcessor processor;
    
    struct ProcessingResult {
        double cpu_time_ms;
        double gpu_time_ms;
        double total_time_ms;
        double throughput_mps;
        size_t total_pixels;
    };

public:

    // Getter per accesso diretto al processor (per benchmark ottimizzato)
    CudaConvolutionProcessor& getProcessor() { return processor; }

private:
    void printDetailedResults(const std::vector<ProcessingResult>& results, 
                            const std::string& test_name);
    double calculateSpeedup(double cpu_time, double gpu_time);
};

#endif // CUDA_IMAGE_PROCESSING_MANAGER_H
