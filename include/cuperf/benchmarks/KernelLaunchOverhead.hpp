#pragma once

#include "cuperf/core/Benchmark.hpp"

namespace cuperf {

__global__ void empty_kernel();

class KernelLaunchOverhead : public Benchmark {
public:
  BenchmarkSpec metadata() const override;
  void setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  void run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  BenchmarkResult run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) override;
  void teardown(BenchmarkContext& ctx) override;

private:
  int block_size_;
};

}
