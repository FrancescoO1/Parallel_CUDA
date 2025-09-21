#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <string>

class Image {
public:
    std::vector<unsigned char> data;
    int width;
    int height;
    int channels;

    // Costruttori
    Image();
    Image(int w, int h, int c);

    // Metodi per I/O
    bool loadFromFile(const std::string& filename);
    bool saveToFile(const std::string& filename) const;

    // Utility
    bool isEmpty() const;
    long getTotalPixels() const;
    void clear();

    // Conversione a grayscale
    Image convertToGrayscale() const;

private:
    // Metodi privati per gestione memoria
    void allocateData();
};

#endif // IMAGE_H