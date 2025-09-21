#include "ConvolutionProcessor.h"
#include <algorithm>
#include <cmath>

// Definizione della matrice kernel statica
constexpr float ConvolutionProcessor::SHARPEN_KERNEL[ConvolutionProcessor::KERNEL_SIZE][ConvolutionProcessor::KERNEL_SIZE];

// Costruttore
ConvolutionProcessor::ConvolutionProcessor() {
    // Nulla da inizializzare per ora
}

// Applica kernel di sharpen
Image ConvolutionProcessor::applySharpenKernel(const Image& input) const {
    return applyKernel3x3(input, SHARPEN_KERNEL);
}

// Applica kernel generico 3x3
Image ConvolutionProcessor::applyKernel3x3(const Image& input, const float kernel[3][3]) const {
    if (input.isEmpty()) {
        return Image();
    }
    
    Image output(input.width, input.height, input.channels);
    
    // Elaborazione pixel per pixel (completamente sequenziale)
    for (int y = 1; y < input.height - 1; y++) {
        for (int x = 1; x < input.width - 1; x++) {
            float result = applyKernelAtPixel(input, x, y, kernel);
            result = clampPixelValue(result);
            
            int output_idx = y * input.width + x;
            output.data[output_idx] = static_cast<unsigned char>(result);
        }
    }
    
    // Copia i bordi dall'immagine originale
    copyBorders(input, output);
    
    return output;
}

// Applica kernel a un singolo pixel
float ConvolutionProcessor::applyKernelAtPixel(const Image& input, int x, int y, const float kernel[3][3]) const {
    float sum = 0.0f;
    
    // Applica kernel 3x3
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int pixel_x = x + kx;
            int pixel_y = y + ky;
            int pixel_idx = pixel_y * input.width + pixel_x;
            
            float pixel_value = static_cast<float>(input.data[pixel_idx]);
            float kernel_value = kernel[ky + 1][kx + 1];
            
            sum += pixel_value * kernel_value;
        }
    }
    
    return sum;
}

// Copia i bordi dall'immagine originale
void ConvolutionProcessor::copyBorders(const Image& input, Image& output) const {
    // Bordi superiore e inferiore
    for (int x = 0; x < input.width; x++) {
        output.data[x] = input.data[x]; // Prima riga
        int bottom_idx = (input.height - 1) * input.width + x;
        output.data[bottom_idx] = input.data[bottom_idx]; // Ultima riga
    }
    
    // Bordi sinistro e destro
    for (int y = 0; y < input.height; y++) {
        int left_idx = y * input.width;
        int right_idx = y * input.width + (input.width - 1);
        output.data[left_idx] = input.data[left_idx]; // Prima colonna
        output.data[right_idx] = input.data[right_idx]; // Ultima colonna
    }
}

// Clamp del valore del pixel tra 0 e 255
float ConvolutionProcessor::clampPixelValue(float value) const {
    return std::max(0.0f, std::min(255.0f, value));
}