#pragma once

#include "cuperf/core/Benchmark.hpp"

namespace cuperf {

class TensorCore : public Benchmark {
public:
  BenchmarkSpec metadata() const override;
  bool is_supported(const GpuInfo& gpu) const override;
  void setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  void run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  BenchmarkResult run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  bool verify_result(BenchmarkContext& ctx) override;
  void teardown(BenchmarkContext& ctx) override;

 private:
  void* d_a_;
  void* d_b_;
  void* d_c_;
  void* d_a_temp_;
  void* d_b_temp_;
  size_t m_;
  size_t n_;
  size_t k_;
  DataType dtype_;
  int gemm_iters_;
};

}
