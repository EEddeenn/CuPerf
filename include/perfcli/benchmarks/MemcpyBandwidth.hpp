#pragma once

#include "perfcli/core/Benchmark.hpp"

namespace perfcli {

class MemcpyBandwidth : public Benchmark {
public:
  BenchmarkSpec metadata() const override;
  bool is_supported(const GpuInfo& gpu) const override;
  void setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  void run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  BenchmarkResult run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  void teardown(BenchmarkContext& ctx) override;

private:
  void* d_ptr_;
  void* h_ptr_;
  size_t size_;
  Direction direction_;
  bool use_pinned_;
  bool use_async_;

  void copy_h2d();
  void copy_d2h();
  void copy_d2d();
};

}
