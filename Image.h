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
    Image(int width, int height, int channels);
    ~Image();

    // Copy constructor e assignment operator
    Image(const Image& other);
    Image& operator=(const Image& other);

    // Move semantics (opzionale per C++11+)
    Image(Image&& other) noexcept = delete;
    Image& operator=(Image&& other) noexcept = delete;

    std::vector<float> toGrayscaleFloat() const;

    static void saveGrayscaleFloat(const std::vector<float>& grayscaleData,
                                  int width, int height, const std::string& filename);

    bool save(const std::string& filename) const;

    int getWidth() const;
    int getHeight() const;
    int getChannels() const;
    unsigned char* getData() const;
};

#endif // IMAGE_H
