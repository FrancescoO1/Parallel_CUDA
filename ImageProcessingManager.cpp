#include "ImageProcessingManager.h"
#include <iostream>
#include <functional>

// Costruttore
ImageProcessingManager::ImageProcessingManager() : default_num_runs(25) {
    printHeader();
}

// Elabora una singola immagine
bool ImageProcessingManager::processSingleImage(const std::string& input_filename, 
                                               const std::string& output_filename, 
                                               int num_runs) {
    // Carica immagine
    Image input_img;
    if (!input_img.loadFromFile(input_filename)) {
        std::cerr << "Impossibile caricare l'immagine: " << input_filename << std::endl;
        return false;
    }
    
    // Converti in grayscale se necessario
    Image grayscale_img = input_img.convertToGrayscale();
    if (grayscale_img.isEmpty()) {
        std::cerr << "Errore nella conversione a grayscale" << std::endl;
        return false;
    }
    
    long total_pixels = grayscale_img.getTotalPixels();
    std::cout << "Pixel totali: " << total_pixels << std::endl;
    
    // Prepara il task di elaborazione
    Image output_img;
    std::function<void()> processing_task = [&]() {
        output_img = processor.applySharpenKernel(grayscale_img);
    };
    
    // Esegui misurazioni multiple
    PerformanceMonitor::Statistics stats = monitor.measureMultipleRuns(
        processing_task, total_pixels, num_runs);
    
    // Stampa statistiche
    monitor.printStatistics(stats);
    
    // Salva immagine risultante
    if (output_img.saveToFile(output_filename)) {
        std::cout << "Immagine salvata: " << output_filename << std::endl;
        return true;
    } else {
        std::cerr << "Errore nel salvataggio di: " << output_filename << std::endl;
        return false;
    }
}

// Elabora multiple immagini con 25 run completi
void ImageProcessingManager::processMultipleImages(const std::vector<std::string>& input_filenames,
                                                  int num_runs) {
    // Carica tutte le immagini prima di iniziare
    std::vector<Image> input_images;
    std::vector<Image> grayscale_images;
    long total_pixels = 0;

    std::cout << "=== Caricamento immagini ===" << std::endl;
    for (size_t i = 0; i < input_filenames.size(); i++) {
        const std::string& filename = input_filenames[i];

        std::cout << "Caricando immagine " << (i + 1) << ": " << filename << std::endl;

        Image input_img;
        if (!input_img.loadFromFile(filename)) {
            std::cerr << "Impossibile caricare l'immagine: " << filename << std::endl;
            continue;
        }

        // Converti in grayscale
        Image grayscale_img = input_img.convertToGrayscale();
        if (grayscale_img.isEmpty()) {
            std::cerr << "Errore nella conversione a grayscale: " << filename << std::endl;
            continue;
        }

        input_images.push_back(std::move(input_img));
        grayscale_images.push_back(std::move(grayscale_img));
        total_pixels += grayscale_images.back().getTotalPixels();

        std::cout << "  Dimensioni: " << grayscale_images.back().width
                  << "x" << grayscale_images.back().height
                  << ", Pixel: " << grayscale_images.back().getTotalPixels() << std::endl;
    }

    if (grayscale_images.empty()) {
        std::cerr << "Nessuna immagine caricata con successo!" << std::endl;
        return;
    }

    std::cout << "\nImmagini caricate: " << grayscale_images.size() << std::endl;
    std::cout << "Pixel totali: " << total_pixels << std::endl;

    // Prepara il task di elaborazione per TUTTE le immagini in sequenza
    std::vector<Image> output_images(grayscale_images.size());

    std::function<void()> batch_processing_task = [&]() {
        // Elabora TUTTE le immagini in sequenza durante ogni run
        for (size_t i = 0; i < grayscale_images.size(); i++) {
            output_images[i] = processor.applySharpenKernel(grayscale_images[i]);
        }
    };

    std::cout << "\n=== Elaborazione Batch (" << num_runs << " run completi) ===" << std::endl;

    // Esegui misurazioni multiple SUL BATCH COMPLETO
    PerformanceMonitor::Statistics stats = monitor.measureMultipleRuns(
        batch_processing_task, total_pixels, num_runs);

    // Stampa statistiche
    monitor.printStatistics(stats);

    // Salva immagini risultanti
    std::cout << "\n=== Salvataggio immagini elaborate ===" << std::endl;
    for (size_t i = 0; i < output_images.size(); i++) {
        std::string output_filename = generateOutputFilename(i + 1);

        if (output_images[i].saveToFile(output_filename)) {
            std::cout << "Immagine salvata: " << output_filename << std::endl;
        } else {
            std::cerr << "Errore nel salvataggio di: " << output_filename << std::endl;
        }
    }
    
    printCompletion();
}

// Configura numero di run
void ImageProcessingManager::setNumRuns(int runs) {
    if (runs > 0) {
        default_num_runs = runs;
    }
}

int ImageProcessingManager::getNumRuns() const {
    return default_num_runs;
}

// Genera nome file di output
std::string ImageProcessingManager::generateOutputFilename(size_t image_index) const {
    return "output_" + std::to_string(image_index) + ".png";
}

// Stampa header del programma
void ImageProcessingManager::printHeader() const {
    std::cout << "=== Kernel Image Processing Sequenziale ===" << std::endl;
    std::cout << "Kernel Sharpen 3x3 applicato" << std::endl << std::endl;
}

// Stampa header per singola immagine
void ImageProcessingManager::printImageHeader(size_t image_index, const std::string& filename) const {
    std::cout << "=== Elaborazione immagine " << image_index << ": " << filename << " ===" << std::endl;
}

// Stampa messaggio di completamento
void ImageProcessingManager::printCompletion() const {
    std::cout << "Elaborazione completata!" << std::endl;
}