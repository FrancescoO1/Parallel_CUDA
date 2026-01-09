#ifndef CONVOLUTION_PROCESSOR_H
#define CONVOLUTION_PROCESSOR_H

#include <vector>

class ConvolutionProcessor {
private:
    static const float sharpen_kernel[9];

public:
    ConvolutionProcessor() = default;
    ~ConvolutionProcessor() = default;

    std::vector<float> applySharpenFilter(const std::vector<float>& image,
                                        size_t width, size_t height) const;

private:
    float applyKernel(const std::vector<float>& image,
                     size_t x, size_t y,
                     size_t width, size_t height) const;
};

#endif
