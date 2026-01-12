#include "perfcli/core/Runner.hpp"

namespace perfcli {

void RunPlan::add_case(const std::string& benchmark_name,
                        const std::map<std::string, std::string>& params) {
  cases.push_back({benchmark_name, params});
}

Runner::Runner(const RunConfig& config)
    : config_(config), device_manager_(DeviceManager::instance()) {
  device_manager_.set_device(config_.device_index);
}

std::vector<BenchmarkResult> Runner::execute_plan(const RunPlan& plan) {
  std::vector<BenchmarkResult> results;
  results.reserve(plan.size());

  for (const auto& config_case : plan.cases) {
    try {
      results.push_back(execute_single_case(config_case));
    } catch (const std::exception& e) {
      BenchmarkResult failure;
      failure.benchmark_name = config_case.benchmark_name;
      failure.params = config_case.params;
      failure.device_index = config_.device_index;
      failure.success = false;
      failure.warnings.push_back(e.what());
      results.push_back(failure);
    }
  }

  return results;
}

BenchmarkContext Runner::create_context() {
  return BenchmarkContext(config_.device_index, config_.stream_count);
}

BenchmarkResult Runner::execute_single_case(const RunPlan::CaseConfig& config_case) {
  BenchmarkResult result;
  result.benchmark_name = config_case.benchmark_name;
  result.params = config_case.params;
  result.device_index = config_.device_index;
  result.success = true;

  auto benchmark = BenchmarkRegistry::instance().create(config_case.benchmark_name);
  if (!benchmark) {
    result.warnings.push_back("Benchmark not found: " + config_case.benchmark_name);
    result.success = false;
    return result;
  }

  auto gpu_info = device_manager_.get_device_info(config_.device_index);
  if (!benchmark->is_supported(gpu_info)) {
    result.warnings.push_back("Benchmark not supported on this GPU");
    result.success = false;
    return result;
  }

  BenchmarkContext ctx = create_context();

  try {
    benchmark->setup(ctx, config_case.params);
    benchmark->run_warmup(ctx, config_case.params);
    result = benchmark->run_measure(ctx, config_case.params);
    result.success = true;

    if (config_.verify_results) {
      bool verified = benchmark->verify_result(ctx);
      if (!verified) {
        result.warnings.push_back("Verification failed");
        result.success = false;
      }
    }

    benchmark->teardown(ctx);
  } catch (const std::exception& e) {
    benchmark->teardown(ctx);
    result.warnings.push_back(std::string("Execution failed: ") + e.what());
    result.success = false;
  }

  return result;
}

}
