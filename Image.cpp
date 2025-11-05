#include "Image.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Costruttore da file
Image::Image(const std::string& filename) {
    int channels_in_file;
    data = stbi_load(filename.c_str(), &width, &height, &channels_in_file, 0);

    if (!data) {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    channels = channels_in_file;

}

// Distruttore
Image::~Image() {
    if (data) {
        stbi_image_free(data);
    }
}

// Costruttore di copia (usato da std::vector)
Image::Image(const Image& other)
    : width(other.width), height(other.height), channels(other.channels) {
    size_t size = width * height * channels;
    data = new unsigned char[size]; // Usa 'new' perché stbi_load usa malloc
    std::copy(other.data, other.data + size, data);
}

// Operatore di assegnazione (usato da std::vector)
Image& Image::operator=(const Image& other) {
    if (this != &other) {
        if (data) {
            stbi_image_free(data); // Usa stbi_image_free se caricato con stbi_load
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

// Conversione a grayscale
std::vector<float> Image::toGrayscaleFloat() const {
    std::vector<float> grayscale(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int pixel_idx = idx * channels;

            if (channels == 1) {
                grayscale[idx] = static_cast<float>(data[pixel_idx]);
            } else if (channels >= 3) {
                // Formula di luminanza standard
                float r = static_cast<float>(data[pixel_idx]);
                float g = static_cast<float>(data[pixel_idx + 1]);
                float b = static_cast<float>(data[pixel_idx + 2]);
                grayscale[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }
    }

    return grayscale;
}

// Metodi getter
int Image::getWidth() const { return width; }
int Image::getHeight() const { return height; }