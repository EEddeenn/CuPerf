#include "perfcli/benchmarks/KernelLaunchOverhead.hpp"
#include "perfcli/util/Error.hpp"
#include "perfcli/cuda/Stream.hpp"
#include "perfcli/core/Statistics.hpp"
#include <format>

namespace perfcli {

__global__ void empty_kernel() {
}

BenchmarkSpec KernelLaunchOverhead::metadata() const {
  BenchmarkSpec spec;
  spec.name = "kernel_launch";
  spec.description = "Measure kernel launch overhead";
  spec.parameters = {"block_size"};
  spec.default_params = {{"block_size", "256"}};
  spec.tags = {BenchmarkTag::Latency};
  return spec;
}

void KernelLaunchOverhead::setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  block_size_ = std::stoi(get_param("block_size", "256"));
}

void KernelLaunchOverhead::run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  for (int i = 0; i < 10; ++i) {
    empty_kernel<<<1, block_size_, 0, ctx.streams[0]->get()>>>();
  }
  ctx.streams[0]->sync();
  CUDA_CHECK_LAST();
}

BenchmarkResult KernelLaunchOverhead::run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  BenchmarkResult result;
  result.benchmark_name = "kernel_launch";
  result.params = params;
  result.device_index = ctx.device_index;

  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  int iters = std::stoi(get_param("iters", "200"));

  EventTimer timer(ctx.streams[0]->get());
  std::vector<double> samples_us;

  for (int i = 0; i < iters; ++i) {
    timer.start();

    empty_kernel<<<1, block_size_, 0, ctx.streams[0]->get()>>>();

    timer.stop();
    timer.sync();
    samples_us.push_back(timer.elapsed_microseconds());
  }

  CUDA_CHECK_LAST();

  auto stats = StatisticsCalculator::calculate(samples_us);

  result.raw_samples_us = samples_us;
  result.median_us = stats.median;
  result.p95_us = stats.p95;
  result.p99_us = stats.p99;
  result.mean_us = stats.mean;
  result.stddev_us = stats.stddev;

  result.metrics["launch_latency_us"] = stats.median;

  return result;
}

void KernelLaunchOverhead::teardown(BenchmarkContext& ctx) {
  ctx.sync_all();
}

}
