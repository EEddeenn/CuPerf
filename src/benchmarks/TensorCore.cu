#include "perfcli/benchmarks/TensorCore.hpp"
#include "perfcli/util/Error.hpp"
#include "perfcli/cuda/Memory.hpp"
#include "perfcli/cuda/Stream.hpp"
#include "perfcli/core/Statistics.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <format>
#include <vector>
#include <cmath>
#include <algorithm>

namespace perfcli {

using namespace nvcuda::wmma;

__inline__ __device__ half fp4_to_fp16(uint8_t packed, bool high_nibble) {
  uint8_t nibble = high_nibble ? (packed >> 4) : (packed & 0x0F);
  int8_t signed_val = static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
  return __int2half_rn(static_cast<int>(signed_val));
}

__global__ void __launch_bounds__(256, 2) tensor_core_fp4_unpack_kernel(
    const uint8_t* __restrict__ a_fp4, const uint8_t* __restrict__ b_fp4,
    half* __restrict__ a_fp16, half* __restrict__ b_fp16,
    size_t m, size_t n, size_t k) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t total_a = m * k;
  size_t total_b = k * n;

  constexpr int kUnrollFactor = 4;

  if (idx * kUnrollFactor < total_a) {
    #pragma unroll
    for (int u = 0; u < kUnrollFactor; ++u) {
      size_t a_idx = idx * kUnrollFactor + u;
      if (a_idx < total_a) {
        uint8_t packed = a_fp4[a_idx / 2];
        a_fp16[a_idx] = fp4_to_fp16(packed, a_idx % 2);
      }
    }
  }

  if (idx * kUnrollFactor < total_b) {
    #pragma unroll
    for (int u = 0; u < kUnrollFactor; ++u) {
      size_t b_idx = idx * kUnrollFactor + u;
      if (b_idx < total_b) {
        uint8_t packed = b_fp4[b_idx / 2];
        b_fp16[b_idx] = fp4_to_fp16(packed, b_idx % 2);
      }
    }
  }
}

__global__ void __launch_bounds__(256, 2) tensor_core_fp4_kernel(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ b,
    half* __restrict__ a_temp, half* __restrict__ b_temp,
    float* __restrict__ c, size_t m, size_t n, size_t k, int iters) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;

  const int lane_id = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int warp_idx_x = warp_idx % 2;
  const int warp_idx_y = warp_idx / 2;

  const int warp_m = blockIdx.y * 2 + warp_idx_y;
  const int warp_n = blockIdx.x * 4 + warp_idx_x;

  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  fill_fragment(c_frag, 0.0f);

  #pragma unroll
  for (int t = 0; t < iters; ++t) {
    for (size_t k_tile = 0; k_tile < k; k_tile += WMMA_K) {
      const size_t a_row = warp_m * WMMA_M + lane_id % 16;
      const size_t a_col = k_tile + (lane_id / 16) * 16;
      const size_t b_row = k_tile + (lane_id / 16) * 16;
      const size_t b_col = warp_n * WMMA_N;

      if (a_row < m && a_col < k) {
        load_matrix_sync(a_frag, a_temp + a_row * k + a_col, k);
      }
      if (b_row < k && b_col < n) {
        load_matrix_sync(b_frag, b_temp + b_row * n + b_col, n);
      }

      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  const size_t c_row = warp_m * WMMA_M;
  const size_t c_col = warp_n * WMMA_N;
  if (c_row < m && c_col < n) {
    store_matrix_sync(c + c_row * n + c_col, c_frag, n, mem_row_major);
  }
}

__global__ void __launch_bounds__(256, 2) tensor_core_fp16_kernel(
    const half* __restrict__ a, const half* __restrict__ b, float* __restrict__ c,
    size_t m, size_t n, size_t k, int iters) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;

  const int lane_id = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int warp_idx_x = warp_idx % 2;
  const int warp_idx_y = warp_idx / 2;

  const int warp_m = blockIdx.y * 2 + warp_idx_y;
  const int warp_n = blockIdx.x * 4 + warp_idx_x;

  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  fill_fragment(c_frag, 0.0f);

  #pragma unroll
  for (int t = 0; t < iters; ++t) {
    for (size_t k_tile = 0; k_tile < k; k_tile += WMMA_K) {
      const size_t a_row = warp_m * WMMA_M + lane_id % 16;
      const size_t a_col = k_tile + (lane_id / 16) * 16;
      const size_t b_row = k_tile + (lane_id / 16) * 16;
      const size_t b_col = warp_n * WMMA_N;

      if (a_row < m && a_col < k) {
        load_matrix_sync(a_frag, a + a_row * k + a_col, k);
      }
      if (b_row < k && b_col < n) {
        load_matrix_sync(b_frag, b + b_row * n + b_col, n);
      }

      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  const size_t c_row = warp_m * WMMA_M;
  const size_t c_col = warp_n * WMMA_N;
  if (c_row < m && c_col < n) {
    store_matrix_sync(c + c_row * n + c_col, c_frag, n, mem_row_major);
  }
}

__global__ void __launch_bounds__(256, 2) tensor_core_bf16_kernel(
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, float* __restrict__ c,
    size_t m, size_t n, size_t k, int iters) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;

  const int lane_id = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int warp_idx_x = warp_idx % 2;
  const int warp_idx_y = warp_idx / 2;

  const int warp_m = blockIdx.y * 2 + warp_idx_y;
  const int warp_n = blockIdx.x * 4 + warp_idx_x;

  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  fill_fragment(c_frag, 0.0f);

  const half* a_half = reinterpret_cast<const half*>(a);
  const half* b_half = reinterpret_cast<const half*>(b);

  #pragma unroll
  for (int t = 0; t < iters; ++t) {
    for (size_t k_tile = 0; k_tile < k; k_tile += WMMA_K) {
      const size_t a_row = warp_m * WMMA_M + lane_id % 16;
      const size_t a_col = k_tile + (lane_id / 16) * 16;
      const size_t b_row = k_tile + (lane_id / 16) * 16;
      const size_t b_col = warp_n * WMMA_N;

      if (a_row < m && a_col < k) {
        load_matrix_sync(a_frag, a_half + a_row * k + a_col, k);
      }
      if (b_row < k && b_col < n) {
        load_matrix_sync(b_frag, b_half + b_row * n + b_col, n);
      }

      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  const size_t c_row = warp_m * WMMA_M;
  const size_t c_col = warp_n * WMMA_N;
  if (c_row < m && c_col < n) {
    store_matrix_sync(c + c_row * n + c_col, c_frag, n, mem_row_major);
  }
}

__global__ void __launch_bounds__(256, 2) tensor_core_int8_kernel(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ c,
    size_t m, size_t n, size_t k, int iters) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;

  const int lane_id = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int warp_idx_x = warp_idx % 2;
  const int warp_idx_y = warp_idx / 2;

  const int warp_m = blockIdx.y * 2 + warp_idx_y;
  const int warp_n = blockIdx.x * 4 + warp_idx_x;

  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, col_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;

  fill_fragment(c_frag, 0);

  #pragma unroll
  for (int t = 0; t < iters; ++t) {
    for (size_t k_tile = 0; k_tile < k; k_tile += WMMA_K) {
      const size_t a_row = warp_m * WMMA_M + lane_id % 16;
      const size_t a_col = k_tile + (lane_id / 16) * 16;
      const size_t b_row = k_tile + (lane_id / 16) * 16;
      const size_t b_col = warp_n * WMMA_N;

      if (a_row < m && a_col < k) {
        load_matrix_sync(a_frag, a + a_row * k + a_col, k);
      }
      if (b_row < k && b_col < n) {
        load_matrix_sync(b_frag, b + b_row * n + b_col, n);
      }

      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  const size_t c_row = warp_m * WMMA_M;
  const size_t c_col = warp_n * WMMA_N;
  if (c_row < m && c_col < n) {
    store_matrix_sync(c + c_row * n + c_col, c_frag, n, mem_row_major);
  }
}

BenchmarkSpec TensorCore::metadata() const {
  BenchmarkSpec spec;
  spec.name = "tensor_core";
  spec.description = "Measure tensor core GEMM performance (WMMA API). BF16 requires CC 8.0+. FP4 uses packed storage with FP16 tensor cores.";
  spec.parameters = {"m", "n", "k", "dtype", "gemm_iters"};
  spec.default_params = {
    {"m", "4096"},
    {"n", "4096"},
    {"k", "4096"},
    {"dtype", "fp16"},
    {"gemm_iters", "1"}
  };
  spec.tags = {BenchmarkTag::Compute};
  spec.supported_types = {DataType::Float16, DataType::BFloat16, DataType::Int8, DataType::Float4};
  return spec;
}

inline half fp4_to_fp16_host(uint8_t packed, bool high_nibble) {
  uint8_t nibble = high_nibble ? (packed >> 4) : (packed & 0x0F);
  int8_t signed_val = static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
  return __float2half_rn(static_cast<float>(signed_val));
}

bool TensorCore::is_supported(const GpuInfo& gpu) const {
  return gpu.supports_tensor_cores();
}

void TensorCore::setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  std::string m_str = get_param("m", "4096");
  std::string n_str = get_param("n", "4096");
  std::string k_str = get_param("k", "4096");
  std::string dtype_str = get_param("dtype", "fp16");
  std::string gemm_iters_str = get_param("gemm_iters", "1");

  m_ = std::stoull(m_str);
  n_ = std::stoull(n_str);
  k_ = std::stoull(k_str);
  dtype_ = string_to_data_type(dtype_str);
  gemm_iters_ = std::stoi(gemm_iters_str);

  size_t elem_size_a, elem_size_b, elem_size_c;
  if (dtype_ == DataType::Float4) {
    elem_size_a = 1;
    elem_size_b = 1;
    elem_size_c = sizeof(float);
  } else if (dtype_ == DataType::Int8) {
    elem_size_a = sizeof(int8_t);
    elem_size_b = sizeof(int8_t);
    elem_size_c = sizeof(int32_t);
  } else if (dtype_ == DataType::BFloat16) {
    elem_size_a = sizeof(__nv_bfloat16);
    elem_size_b = sizeof(__nv_bfloat16);
    elem_size_c = sizeof(float);
  } else {
    elem_size_a = sizeof(half);
    elem_size_b = sizeof(half);
    elem_size_c = sizeof(float);
  }

  CUDA_CHECK(cudaMalloc(&d_a_, m_ * k_ * elem_size_a));
  CUDA_CHECK(cudaMalloc(&d_b_, k_ * n_ * elem_size_b));
  CUDA_CHECK(cudaMalloc(&d_c_, m_ * n_ * elem_size_c));

  if (dtype_ == DataType::Float4) {
    CUDA_CHECK(cudaMalloc(&d_a_temp_, m_ * k_ * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b_temp_, k_ * n_ * sizeof(half)));
  } else {
    d_a_temp_ = nullptr;
    d_b_temp_ = nullptr;
  }

  CUDA_CHECK(cudaMemset(d_a_, 0, m_ * k_ * elem_size_a));
  CUDA_CHECK(cudaMemset(d_b_, 0, k_ * n_ * elem_size_b));
  CUDA_CHECK(cudaMemset(d_c_, 0, m_ * n_ * elem_size_c));

  if (dtype_ == DataType::Float4) {
    CUDA_CHECK(cudaMemset(d_a_temp_, 0, m_ * k_ * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_b_temp_, 0, k_ * n_ * sizeof(half)));
  }
}

void TensorCore::run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  constexpr int kWarpSize = 32;
  constexpr int kWarpM = 8;
  constexpr int kUnpackBlockSize = 256;
  dim3 block(kWarpSize, kWarpM);
  dim3 grid(((n_ + 15) / 16 + 3) / 4, ((m_ + 15) / 16 + 1) / 2);

  for (int i = 0; i < 5; ++i) {
    switch (dtype_) {
      case DataType::Float4:
        tensor_core_fp4_unpack_kernel<<<(m_ * k_ + 255) / 256, kUnpackBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<const uint8_t*>(d_a_), static_cast<const uint8_t*>(d_b_),
            static_cast<half*>(d_a_temp_), static_cast<half*>(d_b_temp_),
            m_, n_, k_);
        tensor_core_fp4_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
            static_cast<const uint8_t*>(d_a_), static_cast<const uint8_t*>(d_b_),
            static_cast<half*>(d_a_temp_), static_cast<half*>(d_b_temp_),
            static_cast<float*>(d_c_), m_, n_, k_, gemm_iters_);
        break;
      case DataType::Int8:
        tensor_core_int8_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
            static_cast<const int8_t*>(d_a_), static_cast<const int8_t*>(d_b_),
            static_cast<int32_t*>(d_c_), m_, n_, k_, gemm_iters_);
        break;
      case DataType::BFloat16:
        tensor_core_bf16_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
            static_cast<const __nv_bfloat16*>(d_a_), static_cast<const __nv_bfloat16*>(d_b_),
            static_cast<float*>(d_c_), m_, n_, k_, gemm_iters_);
        break;
      case DataType::Float16:
      default:
        tensor_core_fp16_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
            static_cast<const half*>(d_a_), static_cast<const half*>(d_b_),
            static_cast<float*>(d_c_), m_, n_, k_, gemm_iters_);
        break;
    }
  }
  ctx.streams[0]->sync();
  CUDA_CHECK_LAST();
}

BenchmarkResult TensorCore::run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  BenchmarkResult result;
  result.benchmark_name = "tensor_core";
  result.params = params;
  result.device_index = ctx.device_index;

  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  int iters = std::stoi(get_param("iters", "200"));

  constexpr int kWarpSize = 32;
  constexpr int kWarpM = 8;
  constexpr int kUnpackBlockSize = 256;
  dim3 block(kWarpSize, kWarpM);
  dim3 grid(((n_ + 15) / 16 + 3) / 4, ((m_ + 15) / 16 + 1) / 2);

  EventTimer timer(ctx.streams[0]->get());
  std::vector<double> samples_us;

  for (int i = 0; i < iters; ++i) {
    timer.start();

    switch (dtype_) {
      case DataType::Float4:
        tensor_core_fp4_unpack_kernel<<<(m_ * k_ + 255) / 256, kUnpackBlockSize, 0, ctx.streams[0]->get()>>>(
            static_cast<const uint8_t*>(d_a_), static_cast<const uint8_t*>(d_b_),
            static_cast<half*>(d_a_temp_), static_cast<half*>(d_b_temp_),
            m_, n_, k_);
        tensor_core_fp4_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
            static_cast<const uint8_t*>(d_a_), static_cast<const uint8_t*>(d_b_),
            static_cast<half*>(d_a_temp_), static_cast<half*>(d_b_temp_),
            static_cast<float*>(d_c_), m_, n_, k_, gemm_iters_);
        break;
      case DataType::Int8:
        tensor_core_int8_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
            static_cast<const int8_t*>(d_a_), static_cast<const int8_t*>(d_b_),
            static_cast<int32_t*>(d_c_), m_, n_, k_, gemm_iters_);
        break;
      case DataType::BFloat16:
        tensor_core_bf16_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
            static_cast<const __nv_bfloat16*>(d_a_), static_cast<const __nv_bfloat16*>(d_b_),
            static_cast<float*>(d_c_), m_, n_, k_, gemm_iters_);
        break;
      case DataType::Float16:
      default:
        tensor_core_fp16_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
            static_cast<const half*>(d_a_), static_cast<const half*>(d_b_),
            static_cast<float*>(d_c_), m_, n_, k_, gemm_iters_);
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

  uint64_t total_ops = 2ULL * m_ * n_ * k_ * gemm_iters_;
  double tflops = static_cast<double>(total_ops) / (stats.median / 1e6) / 1e12;
  result.metrics["tflops"] = tflops;

  return result;
}

void TensorCore::teardown(BenchmarkContext& ctx) {
  if (d_a_) {
    cudaFree(d_a_);
    d_a_ = nullptr;
  }
  if (d_b_) {
    cudaFree(d_b_);
    d_b_ = nullptr;
  }
  if (d_c_) {
    cudaFree(d_c_);
    d_c_ = nullptr;
  }
  if (d_a_temp_) {
    cudaFree(d_a_temp_);
    d_a_temp_ = nullptr;
  }
  if (d_b_temp_) {
    cudaFree(d_b_temp_);
    d_b_temp_ = nullptr;
  }
  ctx.sync_all();
}

bool TensorCore::verify_result(BenchmarkContext& ctx) {
  constexpr size_t verify_m = 128;
  constexpr size_t verify_n = 128;
  constexpr size_t verify_k = 128;

  dim3 block(32, 8);
  dim3 grid(((verify_n + 15) / 16 + 3) / 4, ((verify_m + 15) / 16 + 1) / 2);

  if (dtype_ == DataType::Float4) {
    std::vector<uint8_t> a_h(verify_m * verify_k / 2, 0x77);
    std::vector<uint8_t> b_h(verify_k * verify_n / 2, 0x77);
    std::vector<float> c_h(verify_m * verify_n, 0.0f);

    uint8_t* d_verify_a = nullptr;
    uint8_t* d_verify_b = nullptr;
    half* d_verify_a_temp = nullptr;
    half* d_verify_b_temp = nullptr;
    float* d_verify_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_verify_a, verify_m * verify_k / 2 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_verify_b, verify_k * verify_n / 2 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_verify_a_temp, verify_m * verify_k * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_verify_b_temp, verify_k * verify_n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_verify_c, verify_m * verify_n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_verify_a, a_h.data(), a_h.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_verify_b, b_h.data(), b_h.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    tensor_core_fp4_unpack_kernel<<<(verify_m * verify_k + 255) / 256, 256, 0, ctx.streams[0]->get()>>>(
        d_verify_a, d_verify_b, d_verify_a_temp, d_verify_b_temp, verify_m, verify_n, verify_k);
    tensor_core_fp4_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
        d_verify_a, d_verify_b, d_verify_a_temp, d_verify_b_temp, d_verify_c, verify_m, verify_n, verify_k, 1);

    CUDA_CHECK(cudaMemcpy(c_h.data(), d_verify_c, c_h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    half fp4_val_half = fp4_to_fp16_host(0x77, false);
    float fp4_val = __half2float(fp4_val_half);
    float expected = fp4_val * static_cast<float>(verify_k);
    const float epsilon = 1e-1f;
    bool verified = true;
    for (size_t i = 0; i < c_h.size() && verified; ++i) {
      if (std::abs(c_h[i] - expected) > epsilon) {
        verified = false;
      }
    }

    cudaFree(d_verify_a);
    cudaFree(d_verify_b);
    cudaFree(d_verify_a_temp);
    cudaFree(d_verify_b_temp);
    cudaFree(d_verify_c);
    CUDA_CHECK_LAST();

    return verified;
  } else if (dtype_ == DataType::Int8) {
    std::vector<int8_t> a_h(verify_m * verify_k, 1);
    std::vector<int8_t> b_h(verify_k * verify_n, 1);
    std::vector<int32_t> c_h(verify_m * verify_n, 0);

    int8_t* d_verify_a = nullptr;
    int8_t* d_verify_b = nullptr;
    int32_t* d_verify_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_verify_a, verify_m * verify_k * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_verify_b, verify_k * verify_n * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_verify_c, verify_m * verify_n * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_verify_a, a_h.data(), a_h.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_verify_b, b_h.data(), b_h.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    tensor_core_int8_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
        d_verify_a, d_verify_b, d_verify_c, verify_m, verify_n, verify_k, 1);

    CUDA_CHECK(cudaMemcpy(c_h.data(), d_verify_c, c_h.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int32_t expected = static_cast<int32_t>(verify_k);
    const int32_t epsilon = 2;
    bool verified = true;
    for (size_t i = 0; i < c_h.size() && verified; ++i) {
      if (std::abs(c_h[i] - expected) > epsilon) {
        verified = false;
      }
    }

    cudaFree(d_verify_a);
    cudaFree(d_verify_b);
    cudaFree(d_verify_c);
    CUDA_CHECK_LAST();

    return verified;
  } else if (dtype_ == DataType::Float16) {
    std::vector<half> a_h(verify_m * verify_k, __float2half(1.0f));
    std::vector<half> b_h(verify_k * verify_n, __float2half(1.0f));
    std::vector<float> c_h(verify_m * verify_n, 0.0f);

    half* d_verify_a = nullptr;
    half* d_verify_b = nullptr;
    float* d_verify_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_verify_a, verify_m * verify_k * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_verify_b, verify_k * verify_n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_verify_c, verify_m * verify_n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_verify_a, a_h.data(), a_h.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_verify_b, b_h.data(), b_h.size() * sizeof(half), cudaMemcpyHostToDevice));

    tensor_core_fp16_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
        d_verify_a, d_verify_b, d_verify_c, verify_m, verify_n, verify_k, 1);

    CUDA_CHECK(cudaMemcpy(c_h.data(), d_verify_c, c_h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = static_cast<float>(verify_k);
    const float epsilon = 1e-2f;
    bool verified = true;
    for (size_t i = 0; i < c_h.size() && verified; ++i) {
      if (std::abs(c_h[i] - expected) > epsilon) {
        verified = false;
      }
    }

    cudaFree(d_verify_a);
    cudaFree(d_verify_b);
    cudaFree(d_verify_c);
    CUDA_CHECK_LAST();

    return verified;
  } else if (dtype_ == DataType::BFloat16) {
    std::vector<__nv_bfloat16> a_h(verify_m * verify_k, __float2bfloat16(1.0f));
    std::vector<__nv_bfloat16> b_h(verify_k * verify_n, __float2bfloat16(1.0f));
    std::vector<float> c_h(verify_m * verify_n, 0.0f);

    __nv_bfloat16* d_verify_a = nullptr;
    __nv_bfloat16* d_verify_b = nullptr;
    float* d_verify_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_verify_a, verify_m * verify_k * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_verify_b, verify_k * verify_n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_verify_c, verify_m * verify_n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_verify_a, a_h.data(), a_h.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_verify_b, b_h.data(), b_h.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    tensor_core_bf16_kernel<<<grid, block, 0, ctx.streams[0]->get()>>>(
        d_verify_a, d_verify_b, d_verify_c, verify_m, verify_n, verify_k, 1);

    CUDA_CHECK(cudaMemcpy(c_h.data(), d_verify_c, c_h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = static_cast<float>(verify_k);
    const float epsilon = 1e-2f;
    bool verified = true;
    for (size_t i = 0; i < c_h.size() && verified; ++i) {
      if (std::abs(c_h[i] - expected) > epsilon) {
        verified = false;
      }
    }

    cudaFree(d_verify_a);
    cudaFree(d_verify_b);
    cudaFree(d_verify_c);
    CUDA_CHECK_LAST();

    return verified;
  }

  return false;
}

}
