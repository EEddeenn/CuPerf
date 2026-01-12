#include "cuperf/core/Statistics.hpp"

namespace cuperf {

Statistics StatisticsCalculator::calculate(const std::vector<double>& samples, double trim_percent) {
  return calculate(std::span<const double>(samples), trim_percent);
}

Statistics StatisticsCalculator::calculate(std::span<const double> samples, double trim_percent) {
  Statistics stats{};
  stats.sample_count = samples.size();

  if (samples.empty()) {
    stats.mean = 0.0;
    stats.median = 0.0;
    stats.stddev = 0.0;
    stats.p50 = 0.0;
    stats.p95 = 0.0;
    stats.p99 = 0.0;
    stats.trimmed_mean = 0.0;
    stats.min = 0.0;
    stats.max = 0.0;
    return stats;
  }

  std::vector<double> sorted(samples.begin(), samples.end());
  std::sort(sorted.begin(), sorted.end());

  stats.min = sorted.front();
  stats.max = sorted.back();

  stats.p50 = percentile(std::span<const double>(sorted), 50.0);
  stats.p95 = percentile(std::span<const double>(sorted), 95.0);
  stats.p99 = percentile(std::span<const double>(sorted), 99.0);
  stats.median = stats.p50;

  stats.mean = mean(samples);
  stats.stddev = stddev(samples, stats.mean);
  stats.trimmed_mean = trimmed_mean(std::span<const double>(sorted), trim_percent);

  return stats;
}

double StatisticsCalculator::percentile(const std::vector<double>& sorted, double p) {
  return percentile(std::span<const double>(sorted), p);
}

double StatisticsCalculator::percentile(std::span<const double> sorted, double p) {
  if (sorted.empty()) return 0.0;

  size_t n = sorted.size();
  double idx = p / 100.0 * (n - 1);
  size_t lower = static_cast<size_t>(std::floor(idx));
  size_t upper = static_cast<size_t>(std::ceil(idx));

  if (lower == upper) {
    return sorted[lower];
  }

  double fraction = idx - lower;
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

double StatisticsCalculator::mean(std::span<const double> samples) {
  if (samples.empty()) return 0.0;

  double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  return sum / static_cast<double>(samples.size());
}

double StatisticsCalculator::stddev(std::span<const double> samples, double mean_value) {
  if (samples.size() <= 1) return 0.0;

  double sum_sq = 0.0;
  for (double sample : samples) {
    double diff = sample - mean_value;
    sum_sq += diff * diff;
  }

  return std::sqrt(sum_sq / static_cast<double>(samples.size() - 1));
}

double StatisticsCalculator::trimmed_mean(const std::vector<double>& sorted, double trim_percent) {
  return trimmed_mean(std::span<const double>(sorted), trim_percent);
}

double StatisticsCalculator::trimmed_mean(std::span<const double> sorted, double trim_percent) {
  if (sorted.empty()) return 0.0;

  size_t n = sorted.size();
  size_t trim_count = static_cast<size_t>(std::round(trim_percent * n));

  if (trim_count * 2 >= n) {
    return mean(sorted);
  }

  auto begin_it = sorted.begin() + trim_count;
  auto end_it = sorted.end() - trim_count;
  std::span<const double> trimmed(begin_it, end_it);

  return mean(trimmed);
}

}
