#ifndef IMAGE_PROCESSING_MANAGER_H
#define IMAGE_PROCESSING_MANAGER_H

#include "ConvolutionProcessor.h"
#include <vector>
#include <string>

class ImageProcessingManager {
private:
    ConvolutionProcessor processor;

    struct ProcessingResult {
        double execution_time_ms;
        double throughput_mps;
        size_t total_pixels;
    };

public:

    // Getter per accesso diretto al processor (per benchmark ottimizzato)
    ConvolutionProcessor& getProcessor() { return processor; }

private:

    void printDetailedResults(const std::vector<ProcessingResult>& results,
                            const std::string& test_name);
};

#endif // IMAGE_PROCESSING_MANAGER_H
