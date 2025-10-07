#include "ImageProcessingManager.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

ImageProcessingManager::ProcessingResult
ImageProcessingManager::processImageBenchmark(const Image& image, int run_number) {
    ProcessingResult result;

    // Converti l'immagine in grayscale se necessario
    std::vector<float> grayscale = image.toGrayscaleFloat();
    result.total_pixels = image.getWidth() * image.getHeight();

    // Misurazione del tempo
    auto start = monitor.getCurrentTime();

    std::vector<float> processed = processor.applySharpenFilter(
        grayscale, image.getWidth(), image.getHeight());

    auto end = monitor.getCurrentTime();

    result.execution_time_ms = monitor.calculateDuration(start, end);
    result.throughput_mps = (result.total_pixels / 1e6) / (result.execution_time_ms / 1000.0);

    std::cout << "Run " << std::setw(2) << run_number
              << " - Time: " << std::fixed << std::setprecision(2) << result.execution_time_ms << " ms"
              << " - Throughput: " << std::setprecision(1) << result.throughput_mps << " MP/s"
              << std::endl;

    return result;
}

void ImageProcessingManager::processImages(const std::vector<std::string>& imagePaths) {
    if (imagePaths.empty()) {
        std::cout << "No images to process." << std::endl;
        return;
    }

    std::cout << "\n=== CPU Sequential Batch Image Processing ===" << std::endl;
    std::cout << "Processing " << imagePaths.size() << " images, 25 runs each" << std::endl;

    std::vector<ProcessingResult> all_results;
    size_t total_pixels_per_run = 0;

    // Prima, carica tutte le immagini e calcola i pixel totali
    std::vector<Image> images;
    for (const auto& path : imagePaths) {
        images.emplace_back(path);
        total_pixels_per_run += images.back().getWidth() * images.back().getHeight();
        std::cout << "Loaded: " << path << " ("
                  << images.back().getWidth() << "x" << images.back().getHeight() << ")" << std::endl;
    }

    std::cout << "\nStarting benchmark (25 runs):" << std::endl;

    // Esegui 25 run completi
    for (int run = 1; run <= 25; ++run) {
        double run_total_time = 0.0;

        for (size_t i = 0; i < images.size(); ++i) {
            ProcessingResult result = processImageBenchmark(images[i], run);
            run_total_time += result.execution_time_ms;

            // Salva solo la prima immagine di ogni run per evitare troppi file
            if (i == 0) {
                std::vector<float> grayscale = images[i].toGrayscaleFloat();
                std::vector<float> processed = processor.applySharpenFilter(
                    grayscale, images[i].getWidth(), images[i].getHeight());

                std::string outputPath = "cpu_output_" + std::to_string(run) + ".png";
                Image::saveGrayscaleFloat(processed, images[i].getWidth(),
                                        images[i].getHeight(), outputPath);
            }
        }

        ProcessingResult run_result;
        run_result.execution_time_ms = run_total_time;
        run_result.total_pixels = total_pixels_per_run;
        run_result.throughput_mps = (total_pixels_per_run / 1e6) / (run_total_time / 1000.0);

        all_results.push_back(run_result);

        std::cout << "Run " << run << " complete - Total time: "
                  << std::fixed << std::setprecision(2) << run_total_time << " ms" << std::endl;
    }

    printDetailedResults(all_results, "CPU Sequential Batch Processing");
}

void ImageProcessingManager::processSingleImage(const std::string& imagePath) {
    // Carica e processa semplicemente una singola immagine (senza benchmark interno)
    Image image(imagePath);
    std::vector<float> grayscale = image.toGrayscaleFloat();
    std::vector<float> processed = processor.applySharpenFilter(
        grayscale, image.getWidth(), image.getHeight());
    // Non salviamo l'immagine processata (solo per benchmark)
}

void ImageProcessingManager::printDetailedResults(
    const std::vector<ProcessingResult>& results, const std::string& test_name) {

    if (results.empty()) return;

    // Calcola statistiche
    std::vector<double> times;
    std::vector<double> throughputs;

    for (const auto& result : results) {
        times.push_back(result.execution_time_ms);
        throughputs.push_back(result.throughput_mps);
    }

    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avg_throughput = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / throughputs.size();

    // Calcola varianza e deviazione standard
    double variance = 0.0;
    for (double time : times) {
        variance += std::pow(time - avg_time, 2);
    }
    variance /= times.size();
    double std_dev = std::sqrt(variance);

    // Stampa risultati
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << test_name << " - RESULTS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total pixels processed: " << results[0].total_pixels << std::endl;
    std::cout << "Number of runs: " << results.size() << std::endl;
    std::cout << "\nTiming Results:" << std::endl;
    std::cout << "  Average time: " << avg_time << " ms" << std::endl;
    std::cout << "  Time variance: " << variance << " ms²" << std::endl;
    std::cout << "  Time std dev: " << std_dev << " ms" << std::endl;
    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  Average throughput: " << std::setprecision(1) << avg_throughput << " MP/s" << std::endl;

    std::cout << std::string(60, '=') << std::endl;
}

BenchmarkStats ImageProcessingManager::processImagesAndGetStats(const std::vector<std::string>& imageFiles) {
    const int num_runs = 25;
    std::vector<double> times;
    std::vector<double> throughputs;
    size_t total_pixels = 0;
    std::vector<Image> images;
    for (const auto& path : imageFiles) {
        images.emplace_back(path);
        total_pixels += images.back().getWidth() * images.back().getHeight();
    }
    for (int run = 1; run <= num_runs; ++run) {
        double run_total_time = 0.0;
        for (size_t i = 0; i < images.size(); ++i) {
            auto start = monitor.getCurrentTime();
            std::vector<float> grayscale = images[i].toGrayscaleFloat();
            std::vector<float> processed = processor.applySharpenFilter(grayscale, images[i].getWidth(), images[i].getHeight());
            auto end = monitor.getCurrentTime();
            double time_ms = monitor.calculateDuration(start, end);
            run_total_time += time_ms;
        }
        double throughput = (total_pixels / 1e6) / (run_total_time / 1000.0);
        times.push_back(run_total_time);
        throughputs.push_back(throughput);
    }
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avg_throughput = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / throughputs.size();
    double variance = 0.0;
    for (double t : times) variance += (t - avg_time) * (t - avg_time);
    variance /= times.size();
    double stddev = std::sqrt(variance);
    BenchmarkStats stats{avg_time, stddev, avg_throughput};
    return stats;
}
