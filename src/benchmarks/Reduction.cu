#include "perfcli/benchmarks/Reduction.hpp"
#include "perfcli/util/Error.hpp"
#include "perfcli/cuda/Memory.hpp"
#include "perfcli/cuda/Stream.hpp"
#include "perfcli/core/Statistics.hpp"
#include <format>
#include <vector>
#include <numeric>

namespace perfcli {

__global__ void reduction_kernel(const float* __restrict__ data, float* __restrict__ partial_sums, size_t n) {
  extern __shared__ float s_data[];

  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * (blockDim.x * 2) + tid;

  s_data[tid] = (idx < n) ? data[idx] : 0.0f;
  if (idx + blockDim.x < n) {
    s_data[tid] += data[idx + blockDim.x];
  }
  __syncthreads();

  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = s_data[0];
  }
}

__global__ void final_reduction_kernel(float* __restrict__ partial_sums, float* result, size_t n) {
  size_t idx = threadIdx.x;
  extern __shared__ float s_data[];

  s_data[idx] = (idx < n) ? partial_sums[idx] : 0.0f;
  __syncthreads();

  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (idx < s) {
      s_data[idx] += s_data[idx + s];
    }
    __syncthreads();
  }

  if (idx == 0) {
    *result = s_data[0];
  }
}

BenchmarkSpec Reduction::metadata() const {
  BenchmarkSpec spec;
  spec.name = "reduction";
  spec.description = "Measure reduction performance";
  spec.parameters = {"size", "dtype"};
  spec.default_params = {
    {"size", "1M"},
    {"dtype", "fp32"}
  };
  spec.tags = {BenchmarkTag::Compute};
  spec.supported_types = {DataType::Float32};
  return spec;
}

bool Reduction::is_supported(const GpuInfo& gpu) const {
  return true;
}

void Reduction::setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  std::string size_str = get_param("size", "1M");
  std::string dtype_str = get_param("dtype", "fp32");

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
  num_elements_ = size_ / sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_data_, size_));
  CUDA_CHECK(cudaMemset(d_data_, 1, size_));

  int block_size = 512;
  int grid_size = (num_elements_ + block_size * 2 - 1) / (block_size * 2);

  CUDA_CHECK(cudaMalloc(&d_partial_sums_, grid_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_result_, sizeof(float)));
}

void Reduction::run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  int block_size = 512;
  int grid_size = (num_elements_ + block_size * 2 - 1) / (block_size * 2);

  for (int i = 0; i < 5; ++i) {
    reduction_kernel<<<grid_size, block_size, block_size * sizeof(float), ctx.streams[0]->get()>>>(
        static_cast<float*>(d_data_), static_cast<float*>(d_partial_sums_), num_elements_);

    final_reduction_kernel<<<1, block_size, block_size * sizeof(float), ctx.streams[0]->get()>>>(
        static_cast<float*>(d_partial_sums_), static_cast<float*>(d_result_), grid_size);
  }
  ctx.streams[0]->sync();
  CUDA_CHECK_LAST();
}

BenchmarkResult Reduction::run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  BenchmarkResult result;
  result.benchmark_name = "reduction";
  result.params = params;
  result.device_index = ctx.device_index;

  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  int iters = std::stoi(get_param("iters", "200"));

  int block_size = 512;
  int grid_size = (num_elements_ + block_size * 2 - 1) / (block_size * 2);

  EventTimer timer(ctx.streams[0]->get());
  std::vector<double> samples_us;

  for (int i = 0; i < iters; ++i) {
    timer.start();

    reduction_kernel<<<grid_size, block_size, block_size * sizeof(float), ctx.streams[0]->get()>>>(
        static_cast<float*>(d_data_), static_cast<float*>(d_partial_sums_), num_elements_);

    final_reduction_kernel<<<1, block_size, block_size * sizeof(float), ctx.streams[0]->get()>>>(
        static_cast<float*>(d_partial_sums_), static_cast<float*>(d_result_), grid_size);

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

  double throughput_elements = static_cast<double>(num_elements_) / (stats.median / 1e6);
  result.metrics["throughput_elements_per_sec"] = throughput_elements;

  double bandwidth_gbps = (static_cast<double>(size_) / 1e9) / (stats.median / 1e6);
  result.metrics["bandwidth_gbps"] = bandwidth_gbps;

  return result;
}

void Reduction::teardown(BenchmarkContext& ctx) {
  if (d_data_) {
    cudaFree(d_data_);
    d_data_ = nullptr;
  }
  if (d_partial_sums_) {
    cudaFree(d_partial_sums_);
    d_partial_sums_ = nullptr;
  }
  if (d_result_) {
    cudaFree(d_result_);
    d_result_ = nullptr;
  }
  ctx.sync_all();
}

bool Reduction::verify_result(BenchmarkContext& ctx) {
  const size_t verify_n = 256;

  std::vector<float> input_h(verify_n, 1.0f);
  float expected = std::accumulate(input_h.begin(), input_h.end(), 0.0f);

  float* d_verify_data = nullptr;
  float* d_verify_partial = nullptr;
  float* d_verify_result = nullptr;

  CUDA_CHECK(cudaMalloc(&d_verify_data, verify_n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_verify_partial, 1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_verify_result, sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_verify_data, input_h.data(), verify_n * sizeof(float), cudaMemcpyHostToDevice));

  int block_size = 256;
  int grid_size = (verify_n + block_size * 2 - 1) / (block_size * 2);

  reduction_kernel<<<grid_size, block_size, block_size * sizeof(float), ctx.streams[0]->get()>>>(
      d_verify_data, d_verify_partial, verify_n);

  final_reduction_kernel<<<1, block_size, block_size * sizeof(float), ctx.streams[0]->get()>>>(
      d_verify_partial, d_verify_result, grid_size);

  float result_h;
  CUDA_CHECK(cudaMemcpy(&result_h, d_verify_result, sizeof(float), cudaMemcpyDeviceToHost));

  const float epsilon = 1e-4f;
  bool verified = (std::abs(result_h - expected) < epsilon);

  cudaFree(d_verify_data);
  cudaFree(d_verify_partial);
  cudaFree(d_verify_result);
  CUDA_CHECK_LAST();

  return verified;
}

}
