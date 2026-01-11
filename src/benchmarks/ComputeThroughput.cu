#include "perfcli/benchmarks/ComputeThroughput.hpp"
#include "perfcli/util/Error.hpp"
#include "perfcli/cuda/Memory.hpp"
#include "perfcli/cuda/Stream.hpp"
#include "perfcli/core/Statistics.hpp"
#include <format>
#include <vector>
#include <cmath>
#include <algorithm>

namespace perfcli {

__global__ void fma_kernel(float* __restrict__ data, size_t n, int iters) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float sum = data[idx];
  for (int i = 0; i < iters; ++i) {
    sum = __fmul_rn(sum, 1.0001f);
    sum = __fadd_rn(sum, 0.0001f);
  }
  data[idx] = sum;
}

BenchmarkSpec ComputeThroughput::metadata() const {
  BenchmarkSpec spec;
  spec.name = "compute";
  spec.description = "Measure compute throughput (FMA operations)";
  spec.parameters = {"size", "dtype", "iters"};
  spec.default_params = {
    {"size", "1M"},
    {"dtype", "fp32"},
    {"iters", "10"}
  };
  spec.tags = {BenchmarkTag::Compute};
  spec.supported_types = {DataType::Float32};
  return spec;
}

bool ComputeThroughput::is_supported(const GpuInfo& gpu) const {
  return true;
}

void ComputeThroughput::setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  std::string size_str = get_param("size", "1M");
  std::string dtype_str = get_param("dtype", "fp32");
  std::string iters_str = get_param("iters", "10");

  auto parse_size = [](const std::string& s) -> size_t {
    std::string suffix;
    size_t value = std::stoull(s);
    if (!s.empty()) {
      char last = s.back();
      switch (last) {
        case 'K': case 'k': value *= 1024; break;
        case 'M': case 'm': value *= 1024 * 1024; break;
        case 'G': case 'g': value *= 1024 * 1024 * 1024; break;
      }
    }
    return value;
  };

  size_ = parse_size(size_str);
  dtype_ = string_to_data_type(dtype_str);
  iters_per_launch_ = std::stoi(iters_str);

  CUDA_CHECK(cudaMalloc(&d_data_, size_));
  CUDA_CHECK(cudaMemset(d_data_, 0, size_));
}

void ComputeThroughput::run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  size_t n = size_ / sizeof(float);
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  for (int i = 0; i < 5; ++i) {
    fma_kernel<<<grid_size, block_size, 0, ctx.streams[0]->get()>>>(
        static_cast<float*>(d_data_), n, iters_per_launch_);
  }
  ctx.streams[0]->sync();
  CUDA_CHECK_LAST();
}

BenchmarkResult ComputeThroughput::run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  BenchmarkResult result;
  result.benchmark_name = "compute";
  result.params = params;
  result.device_index = ctx.device_index;

  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  int iters = std::stoi(get_param("iters", "200"));

  size_t n = size_ / sizeof(float);
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  EventTimer timer(ctx.streams[0]->get());
  std::vector<double> samples_us;

  for (int i = 0; i < iters; ++i) {
    timer.start();

    fma_kernel<<<grid_size, block_size, 0, ctx.streams[0]->get()>>>(
        static_cast<float*>(d_data_), n, iters_per_launch_);

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

  double total_ops = static_cast<double>(n) * iters_per_launch_ * 2;
  double gflops = total_ops / (stats.median / 1e6) / 1e9;
  result.metrics["gflops"] = gflops;

  return result;
}

void ComputeThroughput::teardown(BenchmarkContext& ctx) {
  if (d_data_) {
    cudaFree(d_data_);
    d_data_ = nullptr;
  }
  ctx.sync_all();
}

bool ComputeThroughput::verify_result(BenchmarkContext& ctx) {
  const size_t verify_size = 256 * sizeof(float);
  const size_t n = verify_size / sizeof(float);

  std::vector<float> input_h(n, 1.0f);
  std::vector<float> output_h(n);

  float* d_verify = nullptr;
  CUDA_CHECK(cudaMalloc(&d_verify, verify_size));
  CUDA_CHECK(cudaMemcpy(d_verify, input_h.data(), verify_size, cudaMemcpyHostToDevice));

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  fma_kernel<<<grid_size, block_size, 0, ctx.streams[0]->get()>>>(
      d_verify, n, 10);

  CUDA_CHECK(cudaMemcpy(output_h.data(), d_verify, verify_size, cudaMemcpyDeviceToHost));

  float expected = 1.0f;
  for (int i = 0; i < 10; ++i) {
    expected = __builtin_fmaf(expected, 1.0001f, 0.0001f);
  }

  const float epsilon = 1e-4f;
  bool verified = true;
  for (size_t i = 0; i < n && verified; ++i) {
    if (std::abs(output_h[i] - expected) > epsilon) {
      verified = false;
    }
  }

  cudaFree(d_verify);
  CUDA_CHECK_LAST();

  return verified;
}

}
