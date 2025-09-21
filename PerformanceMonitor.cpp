#include "PerformanceMonitor.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// Costruttore
PerformanceMonitor::PerformanceMonitor() {
    // Nulla da inizializzare per ora
}

// Misurazione singola
PerformanceMonitor::PerformanceResult PerformanceMonitor::measureExecution(
    const std::function<void()>& task, long total_pixels) const {
    
    PerformanceResult result;
    result.total_pixels = total_pixels;
    
    // Misurazione del tempo di esecuzione
    auto start_time = std::chrono::high_resolution_clock::now();
    
    task(); // Esegui il task
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Calcola tempo in millisecondi
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.execution_time_ms = duration.count() / 1000.0;
    
    // Calcola throughput
    result.throughput_mpps = calculateThroughput(total_pixels, result.execution_time_ms);
    
    return result;
}

// Misurazione multipla con statistiche
PerformanceMonitor::Statistics PerformanceMonitor::measureMultipleRuns(
    const std::function<void()>& task, long total_pixels, int num_runs) const {
    
    std::vector<double> execution_times;
    std::vector<double> throughputs;
    execution_times.reserve(num_runs);
    throughputs.reserve(num_runs);
    
    std::cout << "Eseguendo " << num_runs << " run completi in batch:" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    for (int run = 0; run < num_runs; run++) {
        PerformanceResult result = measureExecution(task, total_pixels);
        execution_times.push_back(result.execution_time_ms);
        throughputs.push_back(result.throughput_mpps);

        std::cout << "Batch Run " << std::setw(2) << (run + 1)
                  << ": " << std::setw(8) << result.execution_time_ms << " ms, "
                  << std::setw(8) << result.throughput_mpps << " MP/s" << std::endl;
    }
    
    // Calcola statistiche
    Statistics stats;
    stats.num_runs = num_runs;
    
    double variance_time;
    calculateStatistics(execution_times, stats.mean_time_ms, variance_time);
    stats.variance_time_ms = variance_time;
    stats.std_deviation_ms = std::sqrt(variance_time);
    
    // Calcola throughput medio
    double mean_time_seconds = stats.mean_time_ms / 1000.0;
    stats.mean_throughput_mpps = (total_pixels / 1000000.0) / mean_time_seconds;
    
    return stats;
}

// Stampa risultato singolo
void PerformanceMonitor::printResult(const PerformanceResult& result, int run_number) const {
    std::cout << "Run " << std::setw(2) << run_number 
              << ": " << std::setw(8) << result.execution_time_ms << " ms, "
              << std::setw(8) << result.throughput_mpps << " MP/s" << std::endl;
}

// Stampa statistiche
void PerformanceMonitor::printStatistics(const Statistics& stats) const {
    std::cout << "\n--- Statistiche (" << stats.num_runs << " run) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tempo medio: " << std::setw(8) << stats.mean_time_ms << " ms" << std::endl;
    std::cout << "Varianza: " << std::setw(8) << stats.variance_time_ms << " ms²" << std::endl;
    std::cout << "Deviazione standard: " << std::setw(8) << stats.std_deviation_ms << " ms" << std::endl;
    std::cout << "Throughput medio: " << std::setw(8) << stats.mean_throughput_mpps << " MP/s" << std::endl;
}

// Calcola statistiche dai tempi
void PerformanceMonitor::calculateStatistics(const std::vector<double>& times, 
                                            double& mean, double& variance) const {
    if (times.empty()) {
        mean = variance = 0.0;
        return;
    }
    
    // Calcola media
    double sum = 0.0;
    for (double time : times) {
        sum += time;
    }
    mean = sum / times.size();
    
    // Calcola varianza
    double var_sum = 0.0;
    for (double time : times) {
        double diff = time - mean;
        var_sum += diff * diff;
    }
    variance = var_sum / times.size();
}

// Calcola throughput in MegaPixel/s
double PerformanceMonitor::calculateThroughput(long total_pixels, double time_ms) const {
    double time_seconds = time_ms / 1000.0;
    return (total_pixels / 1000000.0) / time_seconds;
}