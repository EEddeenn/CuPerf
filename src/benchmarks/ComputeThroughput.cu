#include "cuperf/benchmarks/ComputeThroughput.hpp"
#include "cuperf/util/Error.hpp"
#include "cuperf/util/Utils.hpp"
#include "cuperf/cuda/Memory.hpp"
#include "cuperf/cuda/Stream.hpp"
#include "cuperf/core/Statistics.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <format>
#include <vector>
#include <cmath>
#include <algorithm>

namespace cuperf {

__global__ void __launch_bounds__(256, 2) fma_kernel(float* __restrict__ data, size_t n, int iters) {
  const size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (base_idx >= n) return;

  float4 sum[2];
  sum[0] = *reinterpret_cast<float4*>(&data[base_idx]);
  sum[1] = *reinterpret_cast<float4*>(&data[base_idx + 4]);
  constexpr float kA = 1.0001f;
  constexpr float kB = 0.0001f;

  #pragma unroll
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      sum[0].x = __fmaf_rn(sum[0].x, kA, kB);
      sum[0].y = __fmaf_rn(sum[0].y, kA, kB);
      sum[0].z = __fmaf_rn(sum[0].z, kA, kB);
      sum[0].w = __fmaf_rn(sum[0].w, kA, kB);
      sum[1].x = __fmaf_rn(sum[1].x, kA, kB);
      sum[1].y = __fmaf_rn(sum[1].y, kA, kB);
      sum[1].z = __fmaf_rn(sum[1].z, kA, kB);
      sum[1].w = __fmaf_rn(sum[1].w, kA, kB);
    }
  }
  *reinterpret_cast<float4*>(&data[base_idx]) = sum[0];
  *reinterpret_cast<float4*>(&data[base_idx + 4]) = sum[1];
}

__global__ void __launch_bounds__(256, 2) fp16_kernel(__half* __restrict__ data, size_t n, int iters) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  __half2 sum2 = *reinterpret_cast<__half2*>(&data[idx * 2]);
  const __half2 kA2 = __float2half2_rn(1.0001f);
  const __half2 kB2 = __float2half2_rn(0.0001f);

  #pragma unroll
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      sum2 = __hfma2(sum2, kA2, kB2);
    }
  }
  *reinterpret_cast<__half2*>(&data[idx * 2]) = sum2;
}

__global__ void __launch_bounds__(256, 2) bf16_kernel(__nv_bfloat16* __restrict__ data, size_t n, int iters) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  __nv_bfloat162 sum2 = *reinterpret_cast<__nv_bfloat162*>(&data[idx * 2]);
  const __nv_bfloat162 kA2 = __float2bfloat162_rn(1.0001f);
  const __nv_bfloat162 kB2 = __float2bfloat162_rn(0.0001f);

  #pragma unroll
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      sum2.x = sum2.x * kA2.x + kB2.x;
      sum2.y = sum2.y * kA2.y + kB2.y;
    }
  }
  *reinterpret_cast<__nv_bfloat162*>(&data[idx * 2]) = sum2;
}

__global__ void __launch_bounds__(256, 2) int8_kernel(int8_t* __restrict__ data, size_t n, int iters) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  int sum = data[idx];
  constexpr int kA = 127;
  constexpr int kB = 1;

  #pragma unroll
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      sum = __dp4a(sum, kA, kB);
    }
  }
  data[idx] = static_cast<int8_t>(sum);
}

__global__ void __launch_bounds__(256, 2) fp4_kernel(uint8_t* __restrict__ data, size_t n, int iters) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  uint8_t packed = data[idx];
  uint8_t nibble0 = packed & 0x0F;
  uint8_t nibble1 = packed >> 4;
  int8_t val0 = static_cast<int8_t>((nibble0 & 0x08) ? (nibble0 | 0xF0) : nibble0);
  int8_t val1 = static_cast<int8_t>((nibble1 & 0x08) ? (nibble1 | 0xF0) : nibble1);
  constexpr int8_t kA = 7;
  constexpr int8_t kB = 1;

  #pragma unroll
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      val0 = (val0 * kA) + kB;
      val1 = (val1 * kA) + kB;
    }
  }

  data[idx] = (static_cast<uint8_t>(val1 & 0x0F) << 4) | (static_cast<uint8_t>(val0 & 0x0F));
}

BenchmarkSpec ComputeThroughput::metadata() const {
  BenchmarkSpec spec;
  spec.name = "compute";
  spec.description = "Measure compute throughput (FMA/DP4A operations)";
  spec.parameters = {"size", "dtype", "iters"};
  spec.default_params = {
    {"size", "1M"},
    {"dtype", "fp32"},
    {"iters", "10"}
  };
  spec.tags = {BenchmarkTag::Compute};
  spec.supported_types = {DataType::Float32, DataType::Float16, DataType::BFloat16, DataType::Int8, DataType::Float4};
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

  size_ = parse_size(size_str);
  dtype_ = string_to_data_type(dtype_str);
  iters_per_launch_ = std::stoi(iters_str);

  CUDA_CHECK(cudaMalloc(&d_data_, size_));
  CUDA_CHECK(cudaMemset(d_data_, 0, size_));
}

void ComputeThroughput::run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  size_t elem_count = size_ / data_type_size(dtype_);
  constexpr int kBlockSize = 256;
  int grid_size;

  if (dtype_ == DataType::Float32) {
    grid_size = ((elem_count / 8) + kBlockSize - 1) / kBlockSize;
  } else if (dtype_ == DataType::Float16 || dtype_ == DataType::BFloat16) {
    grid_size = ((elem_count / 2) + kBlockSize - 1) / kBlockSize;
  } else {
    grid_size = (elem_count + kBlockSize - 1) / kBlockSize;
  }

  for (int i = 0; i < 5; ++i) {
    switch (dtype_) {
      case DataType::Float16:
        fp16_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<__half*>(d_data_), elem_count / 2, iters_per_launch_);
        break;
      case DataType::BFloat16:
        bf16_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<__nv_bfloat16*>(d_data_), elem_count / 2, iters_per_launch_);
        break;
      case DataType::Int8:
        int8_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<int8_t*>(d_data_), elem_count, iters_per_launch_);
        break;
      case DataType::Float4:
        fp4_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<uint8_t*>(d_data_), elem_count, iters_per_launch_);
        break;
      case DataType::Float32:
      default:
        fma_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<float*>(d_data_), elem_count, iters_per_launch_);
        break;
    }
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

  size_t elem_count = size_ / data_type_size(dtype_);
  constexpr int kBlockSize = 256;
  int grid_size;

  if (dtype_ == DataType::Float32) {
    grid_size = ((elem_count / 8) + kBlockSize - 1) / kBlockSize;
  } else if (dtype_ == DataType::Float16 || dtype_ == DataType::BFloat16) {
    grid_size = ((elem_count / 2) + kBlockSize - 1) / kBlockSize;
  } else {
    grid_size = (elem_count + kBlockSize - 1) / kBlockSize;
  }

  EventTimer timer(ctx.streams[0]->get());
  std::vector<double> samples_us;

  for (int i = 0; i < iters; ++i) {
    timer.start();

    switch (dtype_) {
      case DataType::Float16:
        fp16_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<__half*>(d_data_), elem_count / 2, iters_per_launch_);
        break;
      case DataType::BFloat16:
        bf16_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<__nv_bfloat16*>(d_data_), elem_count / 2, iters_per_launch_);
        break;
      case DataType::Int8:
        int8_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<int8_t*>(d_data_), elem_count, iters_per_launch_);
        break;
      case DataType::Float4:
        fp4_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<uint8_t*>(d_data_), elem_count, iters_per_launch_);
        break;
      case DataType::Float32:
      default:
        fma_kernel<<<grid_size, kBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<float*>(d_data_), elem_count, iters_per_launch_);
        break;
    }

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

  switch (dtype_) {
    case DataType::Int8: {
      double total_ops = static_cast<double>(elem_count) * iters_per_launch_ * 16 * 7;
      double tops = total_ops / (stats.median / 1e6) / 1e12;
      result.metrics["tops"] = tops;
      break;
    }
    case DataType::Float4: {
      double total_ops = static_cast<double>(elem_count) * 2 * iters_per_launch_ * 16 * 2;
      double tflops = total_ops / (stats.median / 1e6) / 1e12;
      result.metrics["tflops"] = tflops;
      result.metrics["tops"] = tflops * 1000.0;
      break;
    }
    case DataType::Float16:
    case DataType::BFloat16: {
      double total_ops = static_cast<double>(elem_count / 2) * iters_per_launch_ * 16 * 4;
      double tflops = total_ops / (stats.median / 1e6) / 1e12;
      result.metrics["tflops"] = tflops;
      break;
    }
    case DataType::Float32:
    default: {
      double total_ops = static_cast<double>(elem_count) * iters_per_launch_ * 16 * 2;
      double tflops = total_ops / (stats.median / 1e6) / 1e12;
      result.metrics["tflops"] = tflops;
      double gflops = tflops * 1000.0;
      result.metrics["gflops"] = gflops;
      break;
    }
  }

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
  int grid_size = ((n / 8) + block_size - 1) / block_size;
  fma_kernel<<<grid_size, block_size, 0, ctx.streams[0]->get()>>>(
      d_verify, n, 10);

  CUDA_CHECK(cudaMemcpy(output_h.data(), d_verify, verify_size, cudaMemcpyDeviceToHost));

  float expected = 1.0f;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 16; ++j) {
      expected = __builtin_fmaf(expected, 1.0001f, 0.0001f);
    }
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
