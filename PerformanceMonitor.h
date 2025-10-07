#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <chrono>
#include <vector>
#include <functional>

class PerformanceMonitor {
public:
    struct PerformanceResult {
        double execution_time_ms;
        double throughput_mps;
        size_t total_pixels;
    };

    struct Statistics {
        double average_time_ms;
        double min_time_ms;
        double max_time_ms;
        double variance;
        double std_deviation;
        double average_throughput_mps;
    };

    using TimePoint = std::chrono::high_resolution_clock::time_point;

    PerformanceMonitor() = default;
    ~PerformanceMonitor() = default;

    // Misura l'esecuzione di una singola operazione
    template<typename Func>
    PerformanceResult measureExecution(Func&& operation, size_t total_pixels);

    // Misura multiple esecuzioni e calcola statistiche
    template<typename Func>
    Statistics measureMultipleRuns(Func&& operation, size_t total_pixels, int num_runs = 25);

    // Utility per misurazioni manuali
    TimePoint getCurrentTime() const;
    double calculateDuration(const TimePoint& start, const TimePoint& end) const;

    // Metodi di stampa
    void printResult(const PerformanceResult& result, int run_number = 0) const;
    void printStatistics(const Statistics& stats) const;

private:
    void calculateStatistics(const std::vector<double>& times,
                           const std::vector<double>& throughputs,
                           Statistics& stats) const;
    double calculateThroughput(size_t total_pixels, double time_ms) const;
};

// Implementazione template nel header
template<typename Func>
PerformanceMonitor::PerformanceResult PerformanceMonitor::measureExecution(
    Func&& operation, size_t total_pixels) {

    auto start = getCurrentTime();
    operation();
    auto end = getCurrentTime();

    double time_ms = calculateDuration(start, end);
    double throughput = calculateThroughput(total_pixels, time_ms);

    return {time_ms, throughput, total_pixels};
}

template<typename Func>
PerformanceMonitor::Statistics PerformanceMonitor::measureMultipleRuns(
    Func&& operation, size_t total_pixels, int num_runs) {

    std::vector<double> times;
    std::vector<double> throughputs;
    times.reserve(num_runs);
    throughputs.reserve(num_runs);

    for (int i = 0; i < num_runs; ++i) {
        auto result = measureExecution(operation, total_pixels);
        times.push_back(result.execution_time_ms);
        throughputs.push_back(result.throughput_mps);
    }

    Statistics stats;
    calculateStatistics(times, throughputs, stats);

    return stats;
}

#endif // PERFORMANCE_MONITOR_H
