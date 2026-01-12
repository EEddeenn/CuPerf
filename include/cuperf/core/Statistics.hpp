#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <span>

namespace cuperf {

struct Statistics {
  double mean;
  double median;
  double stddev;
  double p50;
  double p95;
  double p99;
  double trimmed_mean;

  size_t sample_count;
  double min;
  double max;
};

class StatisticsCalculator {
public:
  [[nodiscard]] static Statistics calculate(const std::vector<double>& samples, double trim_percent = 0.05);
  [[nodiscard]] static Statistics calculate(std::span<const double> samples, double trim_percent = 0.05);

  [[nodiscard]] static double percentile(const std::vector<double>& sorted, double p);
  [[nodiscard]] static double percentile(std::span<const double> sorted, double p);
  [[nodiscard]] static double mean(std::span<const double> samples);
  [[nodiscard]] static double stddev(std::span<const double> samples, double mean_value);
  [[nodiscard]] static double trimmed_mean(const std::vector<double>& sorted, double trim_percent);
  [[nodiscard]] static double trimmed_mean(std::span<const double> sorted, double trim_percent);
};

}
