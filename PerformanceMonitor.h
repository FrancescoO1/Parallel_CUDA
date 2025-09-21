#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <vector>
#include <chrono>
#include <functional>

class PerformanceMonitor {
public:
    struct PerformanceResult {
        double execution_time_ms;
        double throughput_mpps;
        long total_pixels;
    };

    struct Statistics {
        double mean_time_ms;
        double variance_time_ms;
        double std_deviation_ms;
        double mean_throughput_mpps;
        int num_runs;
    };

    // Costruttore
    PerformanceMonitor();

    // Misurazione singola
    PerformanceResult measureExecution(const std::function<void()>& task, long total_pixels) const;

    // Misurazione multipla con statistiche
    Statistics measureMultipleRuns(const std::function<void()>& task, long total_pixels, int num_runs) const;

    // Stampa risultati
    void printResult(const PerformanceResult& result, int run_number) const;
    void printStatistics(const Statistics& stats) const;

private:
    // Calcolo statistiche
    void calculateStatistics(const std::vector<double>& times, double& mean, double& variance) const;
    double calculateThroughput(long total_pixels, double time_ms) const;
};

#endif // PERFORMANCE_MONITOR_H