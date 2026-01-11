#include "perfcli/benchmarks/MemcpyBandwidth.hpp"
#include "perfcli/util/Error.hpp"
#include "perfcli/cuda/Memory.hpp"
#include "perfcli/cuda/Stream.hpp"
#include "perfcli/core/Statistics.hpp"
#include <format>

namespace perfcli {

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
  spec.supported_types = {DataType::Float32, DataType::Float16, DataType::Int8};
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
  DataType dtype = string_to_data_type(dtype_str);
  size_t element_size = data_type_size(dtype);
  size_t num_elements = size_ / element_size;

  direction_ = (direction_str == "H2D") ? Direction::HostToDevice :
                (direction_str == "D2H") ? Direction::DeviceToHost :
                Direction::DeviceToDevice;

  use_pinned_ = (pinned_str == "on");
  use_async_ = (async_str == "on");

  HostMemoryType host_type = use_pinned_ ? HostMemoryType::Pinned : HostMemoryType::Pageable;

  d_ptr_ = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ptr_, size_));

  if (direction_ == Direction::HostToDevice || direction_ == Direction::DeviceToHost) {
    h_ptr_ = malloc(size_);
  } else {
    h_ptr_ = nullptr;
  }

  CUDA_CHECK_LAST();
}

void MemcpyBandwidth::run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) {
  int warmup_iters = 10;

  for (int i = 0; i < warmup_iters; ++i) {
    if (direction_ == Direction::HostToDevice) {
      CUDA_CHECK(cudaMemcpyAsync(d_ptr_, h_ptr_, size_, cudaMemcpyHostToDevice, ctx.streams[0]->get()));
    } else if (direction_ == Direction::DeviceToHost) {
      CUDA_CHECK(cudaMemcpyAsync(h_ptr_, d_ptr_, size_, cudaMemcpyDeviceToHost, ctx.streams[0]->get()));
    } else {
      void* d_ptr2;
      CUDA_CHECK(cudaMalloc(&d_ptr2, size_));
      CUDA_CHECK(cudaMemcpyAsync(d_ptr2, d_ptr_, size_, cudaMemcpyDeviceToDevice, ctx.streams[0]->get()));
      cudaFree(d_ptr2);
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
        CUDA_CHECK(cudaMemcpyAsync(d_ptr_, h_ptr_, size_, cudaMemcpyHostToDevice, ctx.streams[0]->get()));
      } else {
        CUDA_CHECK(cudaMemcpy(d_ptr_, h_ptr_, size_, cudaMemcpyHostToDevice));
      }
    } else if (direction_ == Direction::DeviceToHost) {
      if (use_async_) {
        CUDA_CHECK(cudaMemcpyAsync(h_ptr_, d_ptr_, size_, cudaMemcpyDeviceToHost, ctx.streams[0]->get()));
      } else {
        CUDA_CHECK(cudaMemcpy(h_ptr_, d_ptr_, size_, cudaMemcpyDeviceToHost));
      }
    } else {
      void* d_ptr2;
      CUDA_CHECK(cudaMalloc(&d_ptr2, size_));
      CUDA_CHECK(cudaMemcpyAsync(d_ptr2, d_ptr_, size_, cudaMemcpyDeviceToDevice, ctx.streams[0]->get()));
      cudaFree(d_ptr2);
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
  if (d_ptr_) {
    cudaFree(d_ptr_);
    d_ptr_ = nullptr;
  }

  if (h_ptr_) {
    free(h_ptr_);
    h_ptr_ = nullptr;
  }

  ctx.sync_all();
}

}
