#include "ImageProcessingManager.h"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    // Crea il manager per l'elaborazione
    ImageProcessingManager manager;

    // Lista delle immagini da elaborare
    std::vector<std::string> image_files;

    if (argc > 1) {
        // Usa i file passati come argomenti
        for (int i = 1; i < argc; i++) {
            image_files.push_back(argv[i]);
        }
    } else {
        // Usa immagini dal dataset BSD500
        std::string bsd500_path = "/media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/BSDS500/BSDS500/data/images/test/";
        image_files = {
            bsd500_path + "43033.jpg",
            bsd500_path + "41085.jpg",
            bsd500_path + "51084.jpg",
            bsd500_path + "61034.jpg",
            bsd500_path + "65084.jpg",
            bsd500_path + "48017.jpg",
            bsd500_path + "103006.jpg",
            bsd500_path + "109055.jpg",
            bsd500_path + "112090.jpg",
            bsd500_path + "107072.jpg"
        };

        std::cout << "Nessun file specificato. Usando immagini BSD500." << std::endl;
        std::cout << "Uso: " << argv[0] << " <file1> <file2> ... <fileN>" << std::endl;
        std::cout << "Oppure: " << argv[0] << " BSDS500/BSDS500/data/images/test/*.jpg" << std::endl << std::endl;
    }

    // Elabora tutte le immagini
    manager.processMultipleImages(image_files);

    return 0;
}