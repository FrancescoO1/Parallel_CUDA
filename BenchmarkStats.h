#ifndef BENCHMARK_STATS_H
#define BENCHMARK_STATS_H

struct BenchmarkStats {
    double avg_time_ms;
    double stddev_time_ms;
    double avg_throughput_mps;
};

#endif // BENCHMARK_STATS_H

