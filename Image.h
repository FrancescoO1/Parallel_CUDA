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
    explicit Image(const std::string& filename);

    ~Image();

    Image(const Image& other);
    Image& operator=(const Image& other);

    Image(Image&& other) noexcept = delete;
    Image& operator=(Image&& other) noexcept = delete;

    std::vector<float> toGrayscaleFloat() const;

    int getWidth() const;
    int getHeight() const;
};

#endif // IMAGE_H