#include "Image.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

Image::Image(const std::string& filename) {
    int channels_in_file;
    data = stbi_load(filename.c_str(), &width, &height, &channels_in_file, 0);

    if (!data) {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    channels = channels_in_file;

    std::cout << "Loaded image: " << filename
              << " (" << width << "x" << height << ", " << channels << " channels)" << std::endl;
}

Image::Image(int w, int h, int c) : width(w), height(h), channels(c) {
    size_t size = width * height * channels;
    data = new unsigned char[size];
}

Image::~Image() {
    if (data) {
        stbi_image_free(data);
    }
}

Image::Image(const Image& other)
    : width(other.width), height(other.height), channels(other.channels) {
    size_t size = width * height * channels;
    data = new unsigned char[size];
    std::copy(other.data, other.data + size, data);
}

Image& Image::operator=(const Image& other) {
    if (this != &other) {
        if (data) {
            stbi_image_free(data);
        }

        width = other.width;
        height = other.height;
        channels = other.channels;

        size_t size = width * height * channels;
        data = new unsigned char[size];
        std::copy(other.data, other.data + size, data);
    }
    return *this;
}

std::vector<float> Image::toGrayscaleFloat() const {
    std::vector<float> grayscale(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int pixel_idx = idx * channels;

            if (channels == 1) {
                grayscale[idx] = static_cast<float>(data[pixel_idx]);
            } else if (channels >= 3) {
                float r = static_cast<float>(data[pixel_idx]);
                float g = static_cast<float>(data[pixel_idx + 1]);
                float b = static_cast<float>(data[pixel_idx + 2]);
                grayscale[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }
    }

    return grayscale;
}

void Image::saveGrayscaleFloat(const std::vector<float>& grayscaleData,
                              int width, int height, const std::string& filename) {
    std::vector<unsigned char> imageData(width * height);

    for (size_t i = 0; i < grayscaleData.size(); ++i) {
        float value = grayscaleData[i];
        // Clamp tra 0 e 255 e converti a unsigned char
        value = std::max(0.0f, std::min(255.0f, value));
        imageData[i] = static_cast<unsigned char>(value + 0.5f); // Rounded conversion
    }

    if (!stbi_write_png(filename.c_str(), width, height, 1, imageData.data(), width)) {
        throw std::runtime_error("Failed to save image: " + filename);
    }

    std::cout << "Saved image: " << filename << std::endl;
}

bool Image::save(const std::string& filename) const {
    return stbi_write_png(filename.c_str(), width, height, channels, data, width * channels);
}

int Image::getWidth() const { return width; }
int Image::getHeight() const { return height; }
int Image::getChannels() const { return channels; }
unsigned char* Image::getData() const { return data; }
