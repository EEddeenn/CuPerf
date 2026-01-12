#include "cuperf/benchmarks/MemcpyBandwidth.hpp"
#include "cuperf/util/Error.hpp"
#include "cuperf/cuda/Memory.hpp"
#include "cuperf/cuda/Stream.hpp"
#include "cuperf/core/Statistics.hpp"
#include <format>
#include <cstring>
#include <algorithm>

namespace cuperf {

BenchmarkSpec MemcpyBandwidth::metadata() const {
  BenchmarkSpec spec;
  spec.name = "memcpy";
  spec.description = "Measure host-device and device-device memory copy bandwidth";
  spec.parameters = {"size", "direction", "dtype", "pinned", "async"};
  spec.default_params = {
    {"size", "1M"},
    {"direction", "H2D"},
    {"dtype", "fp32"},
    {"pinned", "off"},
    {"async", "off"}
  };
  spec.tags = {BenchmarkTag::Memory};
  spec.supported_types = {DataType::Float32, DataType::Float16, DataType::Int8, DataType::Float4};
  return spec;
}

bool MemcpyBandwidth::is_supported(const GpuInfo& gpu) const {
  return true;
}

void MemcpyBandwidth::setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  auto get_param = [&](const std::string& key, const std::string& default_val) -> std::string {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
  };

  std::string size_str = get_param("size", "1M");
  std::string dtype_str = get_param("dtype", "fp32");
  std::string direction_str = get_param("direction", "H2D");
  std::string pinned_str = get_param("pinned", "off");
  std::string async_str = get_param("async", "off");

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

  direction_ = (direction_str == "H2D") ? Direction::HostToDevice :
                (direction_str == "D2H") ? Direction::DeviceToHost :
                Direction::DeviceToDevice;

  use_pinned_ = (pinned_str == "on");
  use_async_ = (async_str == "on");

  HostMemoryType host_type = use_pinned_ ? HostMemoryType::Pinned : HostMemoryType::Pageable;

  size_t num_elements = size_ / sizeof(uint8_t);
  d_buffer_ = std::make_unique<DeviceBuffer<uint8_t>>(num_elements);

  if (direction_ == Direction::HostToDevice || direction_ == Direction::DeviceToHost) {
    h_buffer_ = std::make_unique<HostBuffer<uint8_t>>(num_elements, host_type);
  } else {
    h_buffer_ = nullptr;
    d_buffer2_ = std::make_unique<DeviceBuffer<uint8_t>>(num_elements);
  }

  CUDA_CHECK_LAST();
}

void MemcpyBandwidth::run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  int warmup_iters = 10;

  for (int i = 0; i < warmup_iters; ++i) {
    if (direction_ == Direction::HostToDevice) {
      CUDA_CHECK(cudaMemcpyAsync(d_buffer_->get(), h_buffer_->get(), size_, cudaMemcpyHostToDevice, ctx.streams[0]->get()));
    } else if (direction_ == Direction::DeviceToHost) {
      CUDA_CHECK(cudaMemcpyAsync(h_buffer_->get(), d_buffer_->get(), size_, cudaMemcpyDeviceToHost, ctx.streams[0]->get()));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(d_buffer2_->get(), d_buffer_->get(), size_, cudaMemcpyDeviceToDevice, ctx.streams[0]->get()));
    }
  }
  ctx.streams[0]->sync();
}

BenchmarkResult MemcpyBandwidth::run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  BenchmarkResult result;
  result.benchmark_name = "memcpy";
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

    if (direction_ == Direction::HostToDevice) {
      if (use_async_) {
        CUDA_CHECK(cudaMemcpyAsync(d_buffer_->get(), h_buffer_->get(), size_, cudaMemcpyHostToDevice, ctx.streams[0]->get()));
      } else {
        CUDA_CHECK(cudaMemcpy(d_buffer_->get(), h_buffer_->get(), size_, cudaMemcpyHostToDevice));
      }
    } else if (direction_ == Direction::DeviceToHost) {
      if (use_async_) {
        CUDA_CHECK(cudaMemcpyAsync(h_buffer_->get(), d_buffer_->get(), size_, cudaMemcpyDeviceToHost, ctx.streams[0]->get()));
      } else {
        CUDA_CHECK(cudaMemcpy(h_buffer_->get(), d_buffer_->get(), size_, cudaMemcpyDeviceToHost));
      }
    } else {
      CUDA_CHECK(cudaMemcpyAsync(d_buffer2_->get(), d_buffer_->get(), size_, cudaMemcpyDeviceToDevice, ctx.streams[0]->get()));
    }

    timer.stop();
    timer.sync();
    samples_us.push_back(timer.elapsed_microseconds());
  }

  auto stats = StatisticsCalculator::calculate(samples_us);

  result.raw_samples_us = samples_us;
  result.median_us = stats.median;
  result.p95_us = stats.p95;
  result.p99_us = stats.p99;
  result.mean_us = stats.mean;
  result.stddev_us = stats.stddev;

  double bandwidth_gbps = (static_cast<double>(size_) / 1e9) / (stats.median / 1e6);
  result.metrics["bandwidth_gbps"] = bandwidth_gbps;

  return result;
}

void MemcpyBandwidth::teardown(BenchmarkContext& ctx) {
  d_buffer_.reset();
  h_buffer_.reset();
  d_buffer2_.reset();
  ctx.sync_all();
}

bool MemcpyBandwidth::verify_result(BenchmarkContext& ctx) {
  const size_t verify_size = std::min(size_, size_t(1024 * 1024));
  std::vector<uint8_t> src_pattern(verify_size);
  std::vector<uint8_t> dst_pattern(verify_size);

  std::memset(src_pattern.data(), 0xAA, verify_size);
  std::memset(dst_pattern.data(), 0x55, verify_size);

  void* d_verify = nullptr;
  CUDA_CHECK(cudaMalloc(&d_verify, verify_size));

  CUDA_CHECK(cudaMemcpy(d_verify, src_pattern.data(), verify_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dst_pattern.data(), d_verify, verify_size, cudaMemcpyDeviceToHost));

  bool verified = (std::memcmp(src_pattern.data(), dst_pattern.data(), verify_size) == 0);

  cudaFree(d_verify);
  CUDA_CHECK_LAST();

  return verified;
}

}
