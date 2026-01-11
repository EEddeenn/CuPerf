#pragma once

#include "perfcli/core/Types.hpp"
#include "perfcli/core/Benchmark.hpp"
#include "perfcli/core/Statistics.hpp"
#include "perfcli/cuda/Device.hpp"
#include <vector>
#include <map>

namespace perfcli {

class RunPlan {
public:
  struct CaseConfig {
    std::string benchmark_name;
    std::map<std::string, std::string> params;
  };

  std::vector<CaseConfig> cases;

  void add_case(const std::string& benchmark_name,
                const std::map<std::string, std::string>& params);

  size_t size() const { return cases.size(); }
};

class Runner {
public:
  explicit Runner(const RunConfig& config);

  std::vector<BenchmarkResult> execute_plan(const RunPlan& plan);

private:
  BenchmarkResult execute_single_case(const RunPlan::CaseConfig& config);

  BenchmarkContext create_context();

  RunConfig config_;
  DeviceManager& device_manager_;
};

}
