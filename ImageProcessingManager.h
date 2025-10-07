#ifndef IMAGE_PROCESSING_MANAGER_H
#define IMAGE_PROCESSING_MANAGER_H

#include "Image.h"
#include "ConvolutionProcessor.h"
#include "PerformanceMonitor.h"
#include "BenchmarkStats.h"
#include <vector>
#include <string>

class ImageProcessingManager {
private:
    ConvolutionProcessor processor;
    PerformanceMonitor monitor;

    struct ProcessingResult {
        double execution_time_ms;
        double throughput_mps;
        size_t total_pixels;
    };

public:
    void processImages(const std::vector<std::string>& imagePaths);
    void processSingleImage(const std::string& imagePath);
    BenchmarkStats processImagesAndGetStats(const std::vector<std::string>& imageFiles);

    // Getter per accesso diretto al processor (per benchmark ottimizzato)
    ConvolutionProcessor& getProcessor() { return processor; }

private:
    ProcessingResult processImageBenchmark(const Image& image, int run_number);
    void printDetailedResults(const std::vector<ProcessingResult>& results,
                            const std::string& test_name);
};

#endif // IMAGE_PROCESSING_MANAGER_H
