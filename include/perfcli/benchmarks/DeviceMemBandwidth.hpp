#pragma once

#include "perfcli/core/Benchmark.hpp"

namespace perfcli {

class DeviceMemBandwidth : public Benchmark {
public:
  BenchmarkSpec metadata() const override;
  bool is_supported(const GpuInfo& gpu) const override;
  void setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  void run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  BenchmarkResult run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  void teardown(BenchmarkContext& ctx) override;

private:
  void* d_src_;
  void* d_dst_;
  size_t size_;
  std::string access_pattern_;

  void run_read_only(BenchmarkContext& ctx, int iters, std::vector<double>& samples_us);
  void run_write_only(BenchmarkContext& ctx, int iters, std::vector<double>& samples_us);
  void run_read_write(BenchmarkContext& ctx, int iters, std::vector<double>& samples_us);
};

}
