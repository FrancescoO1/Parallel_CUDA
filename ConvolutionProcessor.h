#ifndef CONVOLUTION_PROCESSOR_H
#define CONVOLUTION_PROCESSOR_H

#include "Image.h"

class ConvolutionProcessor {
public:
    // Costruttore
    ConvolutionProcessor();

    // Applica kernel di sharpen all'immagine
    Image applySharpenKernel(const Image& input) const;

    // Applica kernel generico 3x3
    Image applyKernel3x3(const Image& input, const float kernel[3][3]) const;

private:
    // Kernel 3x3 di sharpen
    static constexpr int KERNEL_SIZE = 3;
    static constexpr float SHARPEN_KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {
        { 0.0f, -1.0f,  0.0f},
        {-1.0f,  5.0f, -1.0f},
        { 0.0f, -1.0f,  0.0f}
    };

    // Metodi privati per elaborazione
    float applyKernelAtPixel(const Image& input, int x, int y, const float kernel[3][3]) const;
    void copyBorders(const Image& input, Image& output) const;
    float clampPixelValue(float value) const;
};

#endif // CONVOLUTION_PROCESSOR_H