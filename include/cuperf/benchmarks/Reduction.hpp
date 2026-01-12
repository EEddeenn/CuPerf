#pragma once

#include "cuperf/core/Benchmark.hpp"

namespace cuperf {

class Reduction : public Benchmark {
public:
  BenchmarkSpec metadata() const override;
  bool is_supported(const GpuInfo& gpu) const override;
  void setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  void run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  BenchmarkResult run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  bool verify_result(BenchmarkContext& ctx) override;
  void teardown(BenchmarkContext& ctx) override;

private:
  void* d_data_;
  void* d_partial_sums_;
  void* d_result_;
  size_t num_elements_;
  size_t size_;
};

}
