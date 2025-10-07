#include <iostream>
#include <vector>
#include <string>
#include "ImageProcessingManager.h"

int main(int argc, char* argv[]) {
    try {
        ImageProcessingManager manager;

        std::vector<std::string> image_files;

        if (argc > 1) {
            // Usa i file passati come argomenti
            for (int i = 1; i < argc; ++i) {
                image_files.push_back(argv[i]);
            }
        } else {
            // File di default per il test
            image_files = {
                "test_image1.jpg",
                "test_image2.png",
                "test_image3.bmp"
            };
        }

        std::cout << "=== CPU Image Processing Benchmark ===" << std::endl;
        std::cout << "Processing " << image_files.size() << " images..." << std::endl;

        // Singola immagine
        if (!image_files.empty()) {
            std::cout << "\n--- Single Image Test ---" << std::endl;
            manager.processSingleImage(image_files[0]);
        }

        // Multiple images - usa il metodo corretto
        if (image_files.size() > 1) {
            std::cout << "\n--- Multiple Images Test ---" << std::endl;
            manager.processImages(image_files);  // Cambiato da processMultipleImages
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
