#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

class Image {
private:
    int width;
    int height;
    int channels;
    unsigned char* data;

public:
    // Costruttore
    explicit Image(const std::string& filename);

    // Distruttore
    ~Image();

    // Copy constructor e assignment (usati da std::vector)
    Image(const Image& other);
    Image& operator=(const Image& other);

    // Disabilita move semantics
    Image(Image&& other) noexcept = delete;
    Image& operator=(Image&& other) noexcept = delete;

    // Metodo usato dal main per il pre-calcolo
    std::vector<float> toGrayscaleFloat() const;

    // Metodi getter usati dal main
    int getWidth() const;
    int getHeight() const;
};

#endif // IMAGE_H