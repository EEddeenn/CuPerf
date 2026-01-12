#include "cuperf/benchmarks/DeviceMemBandwidth.hpp"
#include "cuperf/util/Error.hpp"
#include "cuperf/cuda/Memory.hpp"
#include "cuperf/cuda/Stream.hpp"
#include "cuperf/core/Statistics.hpp"
#include <format>
#include <cstring>
#include <vector>

namespace cuperf {

__global__ void __launch_bounds__(256, 2) read_kernel(const float* __restrict__ data, size_t n, int iters) {
  const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const size_t stride = (gridDim.x * blockDim.x * 4);
  if (idx >= n) return;

  float4 sum4 = *reinterpret_cast<const float4*>(&data[idx]);

  #pragma unroll
  for (int i = 0; i < iters; ++i) {
    const size_t read_idx = (idx + i * stride) % n;
    const float4 val4 = *reinterpret_cast<const float4*>(&data[read_idx]);
    sum4.x += val4.x;
    sum4.y += val4.y;
    sum4.z += val4.z;
    sum4.w += val4.w;
  }
  (void)sum4;
}

__global__ void __launch_bounds__(256, 2) write_kernel(float* __restrict__ data, size_t n, float4 value) {
  const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx + 4 > n) return;

  *reinterpret_cast<float4*>(&data[idx]) = value;
}

__global__ void __launch_bounds__(256, 2) copy_kernel(float* __restrict__ dst, const float* __restrict__ src, size_t n) {
  const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx + 4 > n) return;

  float4 val4 = *reinterpret_cast<const float4*>(&src[idx]);
  *reinterpret_cast<float4*>(&dst[idx]) = val4;
}

BenchmarkSpec DeviceMemBandwidth::metadata() const {
  BenchmarkSpec spec;
  spec.name = "device_mem";
  spec.description = "Measure device memory bandwidth";
  spec.parameters = {"size", "dtype", "pattern"};
  spec.default_params = {
    {"size", "1M"},
    {"dtype", "fp32"},
    {"pattern", "read_write"}
  };
  spec.tags = {BenchmarkTag::Memory};
  spec.supported_types = {DataType::Float32};
  return spec;
}

bool DeviceMemBandwidth::is_supported(const GpuInfo& gpu) const {
  return true;
}

void DeviceMemBandwidth::setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  std::string size_str = get_param("size", "1M");
  std::string dtype_str = get_param("dtype", "fp32");
  access_pattern_ = get_param("pattern", "read_write");

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

  CUDA_CHECK(cudaMalloc(&d_src_, size_));
  CUDA_CHECK(cudaMalloc(&d_dst_, size_));
}

void DeviceMemBandwidth::run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  std::vector<double> dummy;
  if (access_pattern_ == "read") {
    run_read_only(ctx, 5, dummy);
  } else if (access_pattern_ == "write") {
    run_write_only(ctx, 5, dummy);
  } else {
    run_read_write(ctx, 5, dummy);
  }
}

BenchmarkResult DeviceMemBandwidth::run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  BenchmarkResult result;
  result.benchmark_name = "device_mem";
  result.params = params;
  result.device_index = ctx.device_index;

  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  int iters = std::stoi(get_param("iters", "200"));

  std::vector<double> samples_us;

  if (access_pattern_ == "read") {
    run_read_only(ctx, iters, samples_us);
  } else if (access_pattern_ == "write") {
    run_write_only(ctx, iters, samples_us);
  } else {
    run_read_write(ctx, iters, samples_us);
  }

  auto stats = StatisticsCalculator::calculate(samples_us);

  result.raw_samples_us = samples_us;
  result.median_us = stats.median;
  result.p95_us = stats.p95;
  result.p99_us = stats.p99;
  result.mean_us = stats.mean;
  result.stddev_us = stats.stddev;

  size_t bytes_transferred = size_;
  if (access_pattern_ == "read_write") {
    bytes_transferred *= 2;
  }

  double bandwidth_gbps = (static_cast<double>(bytes_transferred) / 1e9) / (stats.median / 1e6);
  result.metrics["bandwidth_gbps"] = bandwidth_gbps;

  return result;
}

void DeviceMemBandwidth::run_read_only(BenchmarkContext& ctx, int iters, std::vector<double>& samples_us) {
  size_t n = size_ / sizeof(float);
  constexpr int kBlockSize = 256;
  int grid_size = ((n / 4) + kBlockSize - 1) / kBlockSize;

  EventTimer timer(ctx.streams[0]->get());

  for (int i = 0; i < iters; ++i) {
    timer.start();

    read_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
        static_cast<const float*>(d_src_), n, 1);

    timer.stop();
    timer.sync();
    samples_us.push_back(timer.elapsed_microseconds());
  }

  CUDA_CHECK_LAST();
}

void DeviceMemBandwidth::run_write_only(BenchmarkContext& ctx, int iters, std::vector<double>& samples_us) {
  size_t n = size_ / sizeof(float);
  constexpr int kBlockSize = 256;
  constexpr float4 kWriteVal = {1.0f, 1.0f, 1.0f, 1.0f};
  int grid_size = ((n / 4) + kBlockSize - 1) / kBlockSize;

  EventTimer timer(ctx.streams[0]->get());

  for (int i = 0; i < iters; ++i) {
    timer.start();

    write_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
        static_cast<float*>(d_src_), n, kWriteVal);

    timer.stop();
    timer.sync();
    samples_us.push_back(timer.elapsed_microseconds());
  }

  CUDA_CHECK_LAST();
}

void DeviceMemBandwidth::run_read_write(BenchmarkContext& ctx, int iters, std::vector<double>& samples_us) {
  size_t n = size_ / sizeof(float);
  constexpr int kBlockSize = 256;
  int grid_size = ((n / 4) + kBlockSize - 1) / kBlockSize;

  EventTimer timer(ctx.streams[0]->get());

  for (int i = 0; i < iters; ++i) {
    timer.start();

    copy_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
        static_cast<float*>(d_dst_), static_cast<const float*>(d_src_), n);

    timer.stop();
    timer.sync();
    samples_us.push_back(timer.elapsed_microseconds());
  }

  CUDA_CHECK_LAST();
}

void DeviceMemBandwidth::teardown(BenchmarkContext& ctx) {
  if (d_src_) {
    cudaFree(d_src_);
    d_src_ = nullptr;
  }
  if (d_dst_) {
    cudaFree(d_dst_);
    d_dst_ = nullptr;
  }
  ctx.sync_all();
}

bool DeviceMemBandwidth::verify_result(BenchmarkContext& ctx) {
  const size_t verify_size = std::min(size_, size_t(1024 * 1024));
  const size_t n = verify_size / sizeof(float);

  std::vector<float> src_h(n, 1.0f);
  std::vector<float> dst_h(n, 0.0f);

  float* d_verify_src = nullptr;
  float* d_verify_dst = nullptr;

  CUDA_CHECK(cudaMalloc(&d_verify_src, verify_size));
  CUDA_CHECK(cudaMalloc(&d_verify_dst, verify_size));

  CUDA_CHECK(cudaMemcpy(d_verify_src, src_h.data(), verify_size, cudaMemcpyHostToDevice));

  int block_size = 256;
  int grid_size = ((n / 4) + block_size - 1) / block_size;
  copy_kernel<<<grid_size, block_size, 0, ctx.streams[0]->get()>>>(d_verify_dst, d_verify_src, n);

  CUDA_CHECK(cudaMemcpy(dst_h.data(), d_verify_dst, verify_size, cudaMemcpyDeviceToHost));

  bool verified = true;
  for (size_t i = 0; i < n && verified; ++i) {
    if (dst_h[i] != 1.0f) {
      verified = false;
    }
  }

  cudaFree(d_verify_src);
  cudaFree(d_verify_dst);
  CUDA_CHECK_LAST();

  return verified;
}

}
