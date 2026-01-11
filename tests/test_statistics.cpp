#include <gtest/gtest.h>
#include "perfcli/core/Statistics.hpp"
#include <vector>

TEST(StatisticsCalculator, EmptySamples) {
  std::vector<double> samples;
  auto stats = perfcli::StatisticsCalculator::calculate(samples);

  EXPECT_EQ(stats.sample_count, 0);
  EXPECT_EQ(stats.mean, 0.0);
  EXPECT_EQ(stats.median, 0.0);
}

TEST(StatisticsCalculator, SingleSample) {
  std::vector<double> samples = {42.0};
  auto stats = perfcli::StatisticsCalculator::calculate(samples);

  EXPECT_EQ(stats.sample_count, 1);
  EXPECT_DOUBLE_EQ(stats.mean, 42.0);
  EXPECT_DOUBLE_EQ(stats.median, 42.0);
  EXPECT_DOUBLE_EQ(stats.p50, 42.0);
  EXPECT_DOUBLE_EQ(stats.p95, 42.0);
  EXPECT_DOUBLE_EQ(stats.p99, 42.0);
}

TEST(StatisticsCalculator, MultipleSamples) {
  std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0};
  auto stats = perfcli::StatisticsCalculator::calculate(samples);

  EXPECT_EQ(stats.sample_count, 5);
  EXPECT_DOUBLE_EQ(stats.mean, 3.0);
  EXPECT_DOUBLE_EQ(stats.median, 3.0);
  EXPECT_DOUBLE_EQ(stats.min, 1.0);
  EXPECT_DOUBLE_EQ(stats.max, 5.0);
}

TEST(StatisticsCalculator, Percentiles) {
  std::vector<double> samples;
  for (int i = 1; i <= 100; ++i) {
    samples.push_back(static_cast<double>(i));
  }

  auto stats = perfcli::StatisticsCalculator::calculate(samples);

  EXPECT_DOUBLE_EQ(stats.p50, 50.5);
  EXPECT_NEAR(stats.p95, 95.05, 0.1);
  EXPECT_NEAR(stats.p99, 99.01, 0.1);
}

TEST(StatisticsCalculator, TrimmedMean) {
  std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -100.0};
  auto stats = perfcli::StatisticsCalculator::calculate(samples, 0.14);

  EXPECT_NEAR(stats.trimmed_mean, 3.0, 0.5);
  EXPECT_NEAR(stats.mean, 2.143, 0.1);
}

TEST(StatisticsCalculator, StdDev) {
  std::vector<double> samples = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
  auto stats = perfcli::StatisticsCalculator::calculate(samples);

  EXPECT_DOUBLE_EQ(stats.mean, 5.0);
  EXPECT_NEAR(stats.stddev, 2.138, 0.01);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
