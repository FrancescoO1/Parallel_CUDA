#include "Image.h"
#include <iostream>

// Header-only libraries per I/O immagini
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Costruttori
Image::Image() : width(0), height(0), channels(0) {}

Image::Image(int w, int h, int c) : width(w), height(h), channels(c) {
    allocateData();
}

// Carica immagine da file
bool Image::loadFromFile(const std::string& filename) {
    clear();
    
    unsigned char* raw_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!raw_data) {
        std::cerr << "Errore nel caricamento dell'immagine: " << filename << std::endl;
        return false;
    }
    
    std::cout << "Caricata immagine: " << filename 
              << " (" << width << "x" << height 
              << ", " << channels << " canali)" << std::endl;
    
    // Copia i dati nel vettore
    size_t total_size = width * height * channels;
    data.resize(total_size);
    for (size_t i = 0; i < total_size; i++) {
        data[i] = raw_data[i];
    }
    
    stbi_image_free(raw_data);
    return true;
}

// Salva immagine su file
bool Image::saveToFile(const std::string& filename) const {
    if (isEmpty()) {
        std::cerr << "Impossibile salvare immagine vuota" << std::endl;
        return false;
    }
    
    int result = stbi_write_png(filename.c_str(), width, height, 
                               channels, data.data(), width * channels);
    return result != 0;
}

// Verifica se l'immagine è vuota
bool Image::isEmpty() const {
    return data.empty() || width == 0 || height == 0;
}

// Ottieni numero totale di pixel
long Image::getTotalPixels() const {
    return static_cast<long>(width) * height;
}

// Pulisci l'immagine
void Image::clear() {
    data.clear();
    width = height = channels = 0;
}

// Converti in grayscale
Image Image::convertToGrayscale() const {
    if (isEmpty()) {
        return Image();
    }
    
    Image grayscale(width, height, 1);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * channels;
            int dst_idx = y * width + x;
            
            if (channels >= 3) {
                // Conversione RGB a grayscale usando formula standard
                unsigned char r = data[src_idx];
                unsigned char g = data[src_idx + 1];
                unsigned char b = data[src_idx + 2];
                grayscale.data[dst_idx] = static_cast<unsigned char>(
                    0.299f * r + 0.587f * g + 0.114f * b
                );
            } else {
                // Già in grayscale
                grayscale.data[dst_idx] = data[src_idx];
            }
        }
    }
    
    return grayscale;
}

// Alloca memoria per i dati
void Image::allocateData() {
    if (width > 0 && height > 0 && channels > 0) {
        data.resize(width * height * channels);
    }
}