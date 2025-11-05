#ifndef IMAGE_PROCESSING_MANAGER_H
#define IMAGE_PROCESSING_MANAGER_H

#include "ConvolutionProcessor.h"
#include <vector>
#include <string>

class ImageProcessingManager {
private:
    ConvolutionProcessor processor;

public:

    // Getter per accesso diretto al processor
    ConvolutionProcessor& getProcessor() { return processor; }

};

#endif // IMAGE_PROCESSING_MANAGER_H
