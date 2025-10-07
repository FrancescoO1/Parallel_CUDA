#include "PerformanceMonitor.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>

PerformanceMonitor::TimePoint PerformanceMonitor::getCurrentTime() const {
    return std::chrono::high_resolution_clock::now();
}

double PerformanceMonitor::calculateDuration(const TimePoint& start, const TimePoint& end) const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Converti in millisecondi
}

void PerformanceMonitor::printResult(const PerformanceResult& result, int run_number) const {
    std::cout << "Run " << std::setw(2) << run_number
              << " - Time: " << std::fixed << std::setprecision(2) << result.execution_time_ms << " ms"
              << " - Throughput: " << std::setprecision(1) << result.throughput_mps << " MP/s"
              << std::endl;
}

void PerformanceMonitor::printStatistics(const Statistics& stats) const {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "PERFORMANCE STATISTICS" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average time: " << stats.average_time_ms << " ms" << std::endl;
    std::cout << "Min time: " << stats.min_time_ms << " ms" << std::endl;
    std::cout << "Max time: " << stats.max_time_ms << " ms" << std::endl;
    std::cout << "Variance: " << stats.variance << " ms²" << std::endl;
    std::cout << "Std deviation: " << stats.std_deviation << " ms" << std::endl;
    std::cout << "Average throughput: " << std::setprecision(1) << stats.average_throughput_mps << " MP/s" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void PerformanceMonitor::calculateStatistics(const std::vector<double>& times,
                                            const std::vector<double>& throughputs,
                                            Statistics& stats) const {

    stats.average_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    stats.min_time_ms = *std::min_element(times.begin(), times.end());
    stats.max_time_ms = *std::max_element(times.begin(), times.end());
    stats.average_throughput_mps = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / throughputs.size();

    // Calcola varianza
    double sum_squared_diff = 0.0;
    for (double time : times) {
        sum_squared_diff += std::pow(time - stats.average_time_ms, 2);
    }
    stats.variance = sum_squared_diff / times.size();
    stats.std_deviation = std::sqrt(stats.variance);
}

double PerformanceMonitor::calculateThroughput(size_t total_pixels, double time_ms) const {
    // Throughput in megapixel per secondo
    return (total_pixels / 1e6) / (time_ms / 1000.0);
}
