#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
#include "CudaImageProcessingManager.h"

namespace fs = std::filesystem;

std::vector<std::string> getImageFiles(const std::string& directory, int max_images = -1) {
    std::vector<std::string> image_files;

    // Estensioni immagine supportate
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tga"};

    try {
        if (!fs::exists(directory) || !fs::is_directory(directory)) {
            std::cerr << "Directory non trovata: " << directory << std::endl;
            return image_files;
        }

        // Raccoglie tutti i file immagine dalla directory
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                std::string extension = entry.path().extension().string();

                // Converte l'estensione in minuscolo per il confronto
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

                // Controlla se l'estensione è supportata
                if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end()) {
                    image_files.push_back(path);
                }
            }
        }

        // Ordina i file per nome
        std::sort(image_files.begin(), image_files.end());

        // Se specificato un numero massimo, prendi solo i primi N file
        if (max_images > 0 && image_files.size() > static_cast<size_t>(max_images)) {
            image_files.resize(max_images);
        }

    } catch (const fs::filesystem_error& ex) {
        std::cerr << "Errore accesso filesystem: " << ex.what() << std::endl;
    }

    return image_files;
}

void printUsage(const char* program_name) {
    std::cout << "Uso: " << program_name << " [opzioni]" << std::endl;
    std::cout << "Opzioni:" << std::endl;
    std::cout << "  -d <directory>    Directory contenente le immagini (default: ./images)" << std::endl;
    std::cout << "  -n <numero>       Numero massimo di immagini da processare (default: 20)" << std::endl;
    std::cout << "  -s <file>         Processa singola immagine" << std::endl;
    std::cout << "  -h                Mostra questo aiuto" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string images_directory = "/media/francesco/DATA/dev/Clion/Parallel_CUDA_Orlandi_Francesco/archive/blur_dataset_scaled/defocused_blurred";  // Directory default
    int max_images = 150;  // Numero default di immagini
    std::string single_image = "";
    bool process_single = false;

    // Parse degli argomenti della linea di comando
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-d" && i + 1 < argc) {
            images_directory = argv[++i];
        }
        else if (arg == "-n" && i + 1 < argc) {
            max_images = std::stoi(argv[++i]);
            if (max_images <= 0) {
                std::cerr << "Errore: il numero di immagini deve essere maggiore di 0" << std::endl;
                return 1;
            }
        }
        else if (arg == "-s" && i + 1 < argc) {
            single_image = argv[++i];
            process_single = true;
        }
        else {
            std::cerr << "Argomento sconosciuto: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== CUDA Image Processing Benchmark ===" << std::endl;

    CudaImageProcessingManager manager;

    if (process_single) {
        // Modalità singola immagine
        std::cout << "Modalità: Singola immagine" << std::endl;
        std::cout << "File: " << single_image << std::endl;

        if (!fs::exists(single_image)) {
            std::cerr << "Errore: File non trovato: " << single_image << std::endl;
            return 1;
        }

        manager.processSingleImageCUDA(single_image);
    }
    else {
        // Modalità batch
        std::cout << "Modalità: Batch processing" << std::endl;
        std::cout << "Directory: " << images_directory << std::endl;
        std::cout << "Numero massimo immagini: " << max_images << std::endl;

        // Ottieni la lista dei file immagine
        std::vector<std::string> image_files = getImageFiles(images_directory, max_images);

        if (image_files.empty()) {
            std::cerr << "Errore: Nessuna immagine trovata nella directory: "
                      << images_directory << std::endl;
            std::cerr << "Assicurati che la directory esista e contenga file immagine "
                      << "(jpg, jpeg, png, bmp, tga)" << std::endl;
            return 1;
        }

        std::cout << "Trovate " << image_files.size() << " immagini:" << std::endl;
        for (size_t i = 0; i < image_files.size() && i < 5; ++i) {
            std::cout << "  " << fs::path(image_files[i]).filename().string() << std::endl;
        }
        if (image_files.size() > 5) {
            std::cout << "  ... e altre " << (image_files.size() - 5) << " immagini" << std::endl;
        }

        manager.processImagesCUDA(image_files);
    }

    std::cout << "\nProcessing completato!" << std::endl;
    return 0;
}
