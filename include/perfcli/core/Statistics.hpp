#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace perfcli {

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
  static Statistics calculate(const std::vector<double>& samples, double trim_percent = 0.05);

  static double percentile(const std::vector<double>& sorted, double p);
  static double mean(const std::vector<double>& samples);
  static double stddev(const std::vector<double>& samples, double mean_value);
  static double trimmed_mean(const std::vector<double>& sorted, double trim_percent);
};

}
