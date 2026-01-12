#pragma once

#include "perfcli/cli/Args.hpp"
#include "perfcli/core/Types.hpp"
#include "perfcli/core/Runner.hpp"
#include <vector>

namespace perfcli {

class Commands {
public:
  static int execute_info(const Args& args);
  static int execute_list(const Args& args);
  static int execute_run(const Args& args);
  static int execute_selftest(const Args& args);

private:
  static void print_system_info();
  static void print_gpu_info();
  static void print_benchmark_list();

  static RunPlan build_run_plan(const Args& args);
  static std::vector<size_t> parse_sizes(const std::string& sizes_str);
  static std::vector<size_t> parse_size_range(const std::string& range_str);

  static void print_results(const std::vector<BenchmarkResult>& results);
  static void save_results_json(const std::vector<BenchmarkResult>& results,
                                 const std::string& filename);
  static void save_results_csv(const std::vector<BenchmarkResult>& results,
                                const std::string& filename);
};

}
