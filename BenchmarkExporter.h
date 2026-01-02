#ifndef BENCHMARK_EXPORTER_H
#define BENCHMARK_EXPORTER_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip> // Per std::setw, std::setprecision, std::fixed
#include "BenchmarkStats.h" // Assicurati che il percorso sia corretto

// Struttura per contenere i risultati di UN SINGOLO test
struct BenchmarkRun {
    int num_images;
    double total_megapixels;
    BenchmarkStats cpu_stats;
    BenchmarkStats cuda_stats;
};

class BenchmarkExporter {
private:
    std::vector<BenchmarkRun> results;

public:
    BenchmarkExporter() = default;

    // Aggiungi i risultati di un'esecuzione (es. 1, 5, 10 immagini)
    void addRun(int images, double megapixels,
                const BenchmarkStats& cpu, const BenchmarkStats& cuda) {
        results.push_back({images, megapixels, cpu, cuda});
        std::cout << "\n -> Dati per " << images << " immagini registrati." << std::endl;
    }

    // Stampa la tabella riassuntiva a console
    void printConsoleTable() const {
        if (results.empty()) {
            std::cout << "Nessun risultato da stampare." << std::endl;
            return;
        }

        std::cout << "\n\n=================================== RIEPILOGO BENCHMARK FINALE ===================================" << std::endl;
        std::cout << "| Img | MP      | CPU T(ms) | CUDA T(ms) | CPU Thr(MP/s) | CUDA Thr(MP/s) | Speedup |" << std::endl;
        std::cout << "|-----|---------|-----------|------------|---------------|----------------|---------|" << std::endl;

        // Imposta la formattazione per std::cout
        std::cout << std::fixed << std::setprecision(2);

        for (const auto& run : results) {
            double speedup = (run.cuda_stats.avg_time_ms > 0.001) ?
                             (run.cpu_stats.avg_time_ms / run.cuda_stats.avg_time_ms) : 0.0;

            std::cout << "| " << std::setw(3) << run.num_images << " "
                      << "| " << std::setw(7) << run.total_megapixels << " "
                      << "| " << std::setw(9) << run.cpu_stats.avg_time_ms << " "
                      << "| " << std::setw(10) << run.cuda_stats.avg_time_ms << " "
                      << "| " << std::setw(13) << run.cpu_stats.avg_throughput_mps << " "
                      << "| " << std::setw(14) << run.cuda_stats.avg_throughput_mps << " "
                      << "| " << std::setw(6) << speedup << "x |" << std::endl;
        }
        std::cout << "==================================================================================================" << std::endl;
    }

    // Esporta i dati in formato CSV per Excel/Gnuplot/Python
    bool exportToCSV(const std::string& filename) const {
        if (results.empty()) {
            std::cerr << "Nessun risultato da esportare." << std::endl;
            return false;
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Errore: impossibile aprire il file CSV " << filename << std::endl;
            return false;
        }

        // Intestazione (Header)
        file << "NumImages,TotalMegapixels,CPUTimeMs,CPUStdDev,CPUThroughputMPS,";
        file << "CudaTimeMs,CudaStdDev,CudaThroughputMPS,Speedup\n";

        // Dati
        for (const auto& run : results) {
            double speedup = (run.cuda_stats.avg_time_ms > 0.001) ?
                             (run.cpu_stats.avg_time_ms / run.cuda_stats.avg_time_ms) : 0.0;

            file << run.num_images << ","
                 << run.total_megapixels << ","
                 << run.cpu_stats.avg_time_ms << ","
                 << run.cpu_stats.stddev_time_ms << ","
                 << run.cpu_stats.avg_throughput_mps << ","
                 << run.cuda_stats.avg_time_ms << ","
                 << run.cuda_stats.stddev_time_ms << ","
                 << run.cuda_stats.avg_throughput_mps << ","
                 << speedup << "\n";
        }

        file.close();
        std::cout << "\nDati esportati con successo in " << filename << std::endl;
        return true;
    }
};

#endif // BENCHMARK_EXPORTER_H