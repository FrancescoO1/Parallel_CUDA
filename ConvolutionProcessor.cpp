#include "ConvolutionProcessor.h"
#include <algorithm>

// Definizione del kernel statico - KERNEL SHARPEN ULTRA-AGGRESSIVO
const float ConvolutionProcessor::sharpen_kernel[9] = {
    -2.0f, -2.0f, -2.0f,
    -2.0f, 17.0f, -2.0f,
    -2.0f, -2.0f, -2.0f
};

std::vector<float> ConvolutionProcessor::applySharpenFilter(
    const std::vector<float>& image, size_t width, size_t height) const {

    std::vector<float> result(width * height);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t idx = y * width + x;

            // Copia i pixel dei bordi senza applicare la convoluzione
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                result[idx] = image[idx];
            } else {
                result[idx] = applyKernel(image, x, y, width, height);
            }
        }
    }

    return result;
}

float ConvolutionProcessor::applyKernel(const std::vector<float>& image,
                                      size_t x, size_t y,
                                      size_t width, size_t height) const {
    float sum = 0.0f;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            size_t pixel_y = y + ky;
            size_t pixel_x = x + kx;
            size_t pixel_idx = pixel_y * width + pixel_x;
            size_t kernel_idx = (ky + 1) * 3 + (kx + 1);

            sum += image[pixel_idx] * sharpen_kernel[kernel_idx];
        }
    }

    // Clamp il risultato tra 0 e 255
    return std::max(0.0f, std::min(255.0f, sum));
}
