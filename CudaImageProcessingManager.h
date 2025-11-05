#ifndef CUDA_IMAGE_PROCESSING_MANAGER_H
#define CUDA_IMAGE_PROCESSING_MANAGER_H

#include "CudaConvolutionProcessor.h"
#include <vector>
#include <string>

class CudaImageProcessingManager {
private:
    CudaConvolutionProcessor processor;

public:

    // Getter per accesso diretto al processor (per benchmark ottimizzato)
    CudaConvolutionProcessor& getProcessor() { return processor; }
};

#endif // CUDA_IMAGE_PROCESSING_MANAGER_H
