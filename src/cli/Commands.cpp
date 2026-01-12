#include "cuperf/cli/Commands.hpp"
#include "cuperf/cuda/Device.hpp"
#include "cuperf/util/Utils.hpp"
#include <fmt/core.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cmath>

namespace cuperf {

int Commands::execute_info(const Args& args) {
  print_system_info();
  print_gpu_info();
  return 0;
}

int Commands::execute_list(const Args& args) {
  print_benchmark_list();
  return 0;
}

int Commands::execute_run(const Args& args) {
  auto& config = args.config();
  auto plan = build_run_plan(args);

  if (plan.size() == 0) {
    std::cerr << "No benchmark cases to run.\n";
    return 1;
  }

  std::cout << "Running " << plan.size() << " benchmark case(s)...\n\n";

  Runner runner(config);
  auto results = runner.execute_plan(plan);

  print_results(results);

  if (!config.output_json_file.empty()) {
    save_results_json(results, config.output_json_file);
  }

#ifdef CUPERF_ENABLE_CSV
  if (!config.output_csv_file.empty()) {
    save_results_csv(results, config.output_csv_file);
  }
#endif

  bool all_success = std::all_of(results.begin(), results.end(),
                                   [](const BenchmarkResult& r) { return r.success; });
  return all_success ? 0 : 1;
}

void Commands::print_system_info() {
  std::cout << "=== System Information ===\n";
  std::cout << "CUDA Runtime: " << CUDART_VERSION / 1000 << "."
            << (CUDART_VERSION % 100) / 10 << "\n";
  std::cout << "\n";
}

void Commands::print_gpu_info() {
  auto& device_manager = DeviceManager::instance();
  auto gpus = device_manager.enumerate_devices();

  std::cout << "=== GPU Information ===\n";
  for (const auto& gpu : gpus) {
    std::cout << "GPU " << gpu.device_index << ": " << gpu.name << "\n";
    std::cout << "  Compute Capability: " << gpu.compute_capability << "\n";
    std::cout << "  SM Count: " << gpu.sm_count << "\n";
    std::cout << "  Total Memory: " << gpu.total_memory_mb << " MB\n";
    std::cout << "  UUID: " << gpu.uuid << "\n\n";
  }
}

void Commands::print_benchmark_list() {
  auto& registry = BenchmarkRegistry::instance();
  auto specs = registry.get_all_metadata();

  std::cout << "=== Available Benchmarks ===\n\n";

  for (const auto& spec : specs) {
    std::cout << spec.name << ": " << spec.description << "\n";

    if (!spec.parameters.empty()) {
      std::cout << "  Parameters: ";
      for (size_t i = 0; i < spec.parameters.size(); ++i) {
        std::cout << spec.parameters[i];
        if (i < spec.parameters.size() - 1) std::cout << ", ";
      }
      std::cout << "\n";
    }

    if (!spec.tags.empty()) {
      std::cout << "  Tags: ";
      for (size_t i = 0; i < spec.tags.size(); ++i) {
        switch (spec.tags[i]) {
          case BenchmarkTag::Memory: std::cout << "memory"; break;
          case BenchmarkTag::Compute: std::cout << "compute"; break;
          case BenchmarkTag::Latency: std::cout << "latency"; break;
          case BenchmarkTag::MultiGpu: std::cout << "multi-gpu"; break;
        }
        if (i < spec.tags.size() - 1) std::cout << ", ";
      }
      std::cout << "\n";
    }

    std::cout << "\n";
  }
}

RunPlan Commands::build_run_plan(const Args& args) {
  RunPlan plan;
  auto& registry = BenchmarkRegistry::instance();
  auto& config = args.config();

  std::vector<std::string> benchmark_names;

  if (!config.benchmark_filter.empty()) {
    std::istringstream iss(config.benchmark_filter);
    std::string name;
    while (std::getline(iss, name, ',')) {
      benchmark_names.push_back(name);
    }
  } else if (!config.tag_filter.empty()) {
    BenchmarkTag tag;
    if (config.tag_filter == "memory") tag = BenchmarkTag::Memory;
    else if (config.tag_filter == "compute") tag = BenchmarkTag::Compute;
    else if (config.tag_filter == "latency") tag = BenchmarkTag::Latency;
    else if (config.tag_filter == "multi-gpu") tag = BenchmarkTag::MultiGpu;

    benchmark_names = registry.filter_by_tag(tag);
  } else {
    benchmark_names = registry.list_benchmarks();
  }

  std::vector<size_t> sizes;
  if (config.extra_params.find("sizes") != config.extra_params.end()) {
    sizes = parse_sizes(config.extra_params.at("sizes"));
  } else if (config.extra_params.find("sizes_range") != config.extra_params.end()) {
    sizes = parse_size_range(config.extra_params.at("sizes_range"));
  }

  for (const auto& name : benchmark_names) {
    if (!registry.exists(name)) continue;

    std::map<std::string, std::string> params;
    for (const auto& [key, value] : config.extra_params) {
      if (key != "sizes" && key != "sizes_range") {
        params[key] = value;
      }
    }

    if (sizes.empty()) {
      plan.add_case(name, params);
    } else {
      for (auto size : sizes) {
        params["size"] = std::to_string(size);
        plan.add_case(name, params);
      }
    }
  }

  return plan;
}

std::vector<size_t> Commands::parse_sizes(const std::string& sizes_str) {
  std::vector<size_t> sizes;
  std::istringstream iss(sizes_str);
  std::string token;
  while (std::getline(iss, token, ',')) {
    sizes.push_back(parse_size(token));
  }
  return sizes;
}

std::vector<size_t> Commands::parse_size_range(const std::string& range_str) {
  std::vector<size_t> sizes;
  std::istringstream iss(range_str);
  std::string token;

  std::vector<std::string> parts;
  while (std::getline(iss, token, ':')) {
    parts.push_back(token);
  }

  if (parts.size() != 3) {
    std::cerr << "Invalid size range format. Expected: start:stop:factor\n";
    return sizes;
  }

  size_t start = parse_size(parts[0]);
  size_t stop = parse_size(parts[1]);
  double factor = std::stod(parts[2]);

  for (size_t size = start; size <= stop; size = static_cast<size_t>(size * factor)) {
    sizes.push_back(size);
  }

  return sizes;
}

void Commands::print_results(const std::vector<BenchmarkResult>& results) {
  if (results.empty()) return;

  std::cout << "\n=== Results ===\n\n";

  std::cout << std::left << std::setw(20) << "Benchmark"
            << std::setw(12) << "Median"
            << std::setw(12) << "P95"
            << std::setw(12) << "Mean"
            << "\n";
  std::cout << std::string(56, '-') << "\n";

  for (const auto& result : results) {
    if (!result.success) {
      std::cout << std::left << std::setw(20) << result.benchmark_name
                << std::setw(12) << "FAILED"
                << "\n";
      for (const auto& warning : result.warnings) {
        std::cout << "  Warning: " << warning << "\n";
      }
      continue;
    }

    std::cout << std::left << std::setw(20) << result.benchmark_name
              << std::setw(12) << fmt::format("{:.2f} µs", result.median_us)
              << std::setw(12) << fmt::format("{:.2f} µs", result.p95_us)
              << std::setw(12) << fmt::format("{:.2f} µs", result.mean_us)
              << "\n";
  }
}

void Commands::save_results_json(const std::vector<BenchmarkResult>& results,
                                  const std::string& filename) {
  auto& device_manager = DeviceManager::instance();

  std::string json = "{\n";

  SystemInfo info;
  info.cuda_runtime_version = std::to_string(CUDART_VERSION / 1000) + "." +
                                std::to_string((CUDART_VERSION % 100) / 10);
  info.gpus = device_manager.enumerate_devices();

  json += "  \"system_info\": " + info.to_json() + ",\n\n";

  json += "  \"results\": [\n";

  for (size_t i = 0; i < results.size(); ++i) {
    json += "    " + results[i].to_json();
    if (i < results.size() - 1) json += ",";
    json += "\n";
  }

  json += "  ]\n";
  json += "}\n";

  if (filename == "-") {
    std::cout << json;
  } else {
    std::ofstream file(filename);
    file << json;
    std::cout << "Results saved to: " << filename << "\n";
  }
}

void Commands::save_results_csv(const std::vector<BenchmarkResult>& results,
                                 const std::string& filename) {
  std::ofstream file(filename);

  std::vector<std::string> metric_keys;
  for (const auto& result : results) {
    if (!result.metrics.empty()) {
      for (const auto& [key, _] : result.metrics) {
        if (std::find(metric_keys.begin(), metric_keys.end(), key) == metric_keys.end()) {
          metric_keys.push_back(key);
        }
      }
    }
  }

  file << "benchmark,device,median_us,p95_us,p99_us,mean_us,stddev_us";
  for (const auto& key : metric_keys) {
    file << "," << key;
  }
  file << "\n";

  for (const auto& result : results) {
    file << result.benchmark_name << ","
         << result.device_index << ","
         << result.median_us << ","
         << result.p95_us << ","
         << result.p99_us << ","
         << result.mean_us << ","
         << result.stddev_us;

    for (const auto& key : metric_keys) {
      auto it = result.metrics.find(key);
      if (it != result.metrics.end()) {
        file << "," << it->second;
      } else {
        file << ",";
      }
    }

    file << "\n";
  }
}

int Commands::execute_selftest(const Args& args) {
  std::cout << "=== Running smoke tests ===\n\n";

  RunConfig config;
  config.device_index = 0;
  config.verify_results = true;
  config.warmup_iterations = 2;
  config.measured_iterations = 5;

  RunPlan plan;
  std::map<std::string, std::string> params;

  params["size"] = "1K";
  params["dtype"] = "fp32";
  params["direction"] = "H2D";
  params["pinned"] = "off";
  params["async"] = "off";
  plan.add_case("memcpy", params);

  params["size"] = "1K";
  params["pattern"] = "read_write";
  plan.add_case("device_mem", params);

  params["size"] = "1K";
  params["iters"] = "10";
  plan.add_case("compute", params);

  params["size"] = "1K";
  plan.add_case("reduction", params);

  params["block_size"] = "256";
  plan.add_case("kernel_launch", params);

  Runner runner(config);
  auto results = runner.execute_plan(plan);

  print_results(results);

  bool all_passed = std::all_of(results.begin(), results.end(),
                                   [](const BenchmarkResult& r) { return r.success; });

  std::cout << "\n=== Test Summary ===\n";
  if (all_passed) {
    std::cout << "✓ All smoke tests passed\n";
  } else {
    std::cout << "✗ Some smoke tests failed\n";
  }

  return all_passed ? 0 : 1;
}

}
