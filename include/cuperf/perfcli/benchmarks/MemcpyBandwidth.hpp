#pragma once

#include "perfcli/core/Benchmark.hpp"
#include "perfcli/cuda/Memory.hpp"
#include <memory>

namespace perfcli {

class MemcpyBandwidth : public Benchmark {
public:
  BenchmarkSpec metadata() const override;
  bool is_supported(const GpuInfo& gpu) const override;
  void setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  void run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  BenchmarkResult run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  bool verify_result(BenchmarkContext& ctx) override;
  void teardown(BenchmarkContext& ctx) override;

private:
  std::unique_ptr<DeviceBuffer<uint8_t>> d_buffer_;
  std::unique_ptr<HostBuffer<uint8_t>> h_buffer_;
  std::unique_ptr<DeviceBuffer<uint8_t>> d_buffer2_;
  size_t size_;
  Direction direction_;
  bool use_pinned_;
  bool use_async_;
};

}
