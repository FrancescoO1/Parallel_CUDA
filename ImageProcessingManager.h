#ifndef IMAGE_PROCESSING_MANAGER_H
#define IMAGE_PROCESSING_MANAGER_H

#include "Image.h"
#include "ConvolutionProcessor.h"
#include "PerformanceMonitor.h"
#include <vector>
#include <string>

class ImageProcessingManager {
public:
    // Costruttore
    ImageProcessingManager();

    // Elabora una singola immagine
    bool processSingleImage(const std::string& input_filename,
                           const std::string& output_filename,
                           int num_runs = 25);

    // Elabora multiple immagini
    void processMultipleImages(const std::vector<std::string>& input_filenames,
                              int num_runs = 25);

    // Configurazione
    void setNumRuns(int runs);
    int getNumRuns() const;

private:
    ConvolutionProcessor processor;
    PerformanceMonitor monitor;
    int default_num_runs;

    // Metodi di utilità
    std::string generateOutputFilename(size_t image_index) const;
    void printHeader() const;
    void printImageHeader(size_t image_index, const std::string& filename) const;
    void printCompletion() const;
};

#endif // IMAGE_PROCESSING_MANAGER_H