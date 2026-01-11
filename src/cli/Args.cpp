#include "perfcli/cli/Args.hpp"
#include <CLI/CLI.hpp>
#include <iostream>

namespace perfcli {

Args Args::parse(int argc, char* argv[]) {
  Args args;

  CLI::App app{"CUDA Performance Benchmarking CLI Tool", "perfcli"};

  app.set_help_flag("-h,--help", "Display this help and exit");
  app.set_version_flag("-v,--version", "0.1.0");

  auto info_cmd = app.add_subcommand("info", "Display GPU and system information");
  info_cmd->callback([&] { args.command_ = Command::Info; });

  auto list_cmd = app.add_subcommand("list", "List available benchmarks");
  list_cmd->callback([&] { args.command_ = Command::List; });

  auto run_cmd = app.add_subcommand("run", "Run benchmarks");
  run_cmd->callback([&] { args.command_ = Command::Run; });

  auto selftest_cmd = app.add_subcommand("selftest", "Run basic smoke tests");
  selftest_cmd->callback([&] { args.command_ = Command::Selftest; });

  auto& config = args.config_;

  run_cmd->add_option("-d,--device", config.device_index, "GPU device index")
      ->check(CLI::Range(0, 255));

  run_cmd->add_option("--warmup", config.warmup_iterations, "Number of warmup iterations");

  run_cmd->add_option("--iters", config.measured_iterations, "Number of measured iterations");

  run_cmd->add_option("--samples", config.sample_count, "Number of sample runs per case");

  run_cmd->add_option("--streams", config.stream_count, "Number of CUDA streams to use")
      ->check(CLI::Range(1, 32));

  run_cmd->add_flag("--pinned", config.use_pinned_memory, "Use pinned host memory");

  run_cmd->add_flag("--async", config.use_async_copies, "Use async copies");

  run_cmd->add_flag("--verify", config.verify_results, "Verify benchmark results");

  run_cmd->add_option("--json", config.output_json_file, "Output JSON to file (or '-' for stdout)");

  run_cmd->add_option("--csv", config.output_csv_file, "Output CSV to file");

  run_cmd->add_option("--tag", config.tag_filter, "Filter benchmarks by tag (memory|compute|latency|multi-gpu)");

  std::string dtype_str = "fp32";
  run_cmd->add_option("--dtype", dtype_str, "Data type (fp32|fp16|bf16|int8|int32)")
      ->check(CLI::IsMember({"fp32", "fp16", "bf16", "int8", "int32"}));

  std::string direction_str = "H2D";
  run_cmd->add_option("--direction", direction_str, "Copy direction (H2D|D2H|D2D)")
      ->check(CLI::IsMember({"H2D", "D2H", "D2D"}));

  std::vector<std::string> size_list;
  run_cmd->add_option("--sizes", size_list, "Specific sizes to test (e.g., 1K,4M,2G)");

  std::string size_range;
  run_cmd->add_option("--sizes-range", size_range, "Size range (e.g., 1K:1G:2x for geometric progression)");

  run_cmd->add_option("benchmarks", args.benchmarks_, "Benchmark names to run (all if not specified)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    int exit_code = app.exit(e);
    std::exit(exit_code);
  }

  if (app.got_subcommand("run")) {
    config.extra_params["dtype"] = dtype_str;
    config.extra_params["direction"] = direction_str;

    if (!size_list.empty()) {
      config.extra_params["sizes"] = "";
      for (const auto& s : size_list) {
        config.extra_params["sizes"] += s + ",";
      }
      if (!config.extra_params["sizes"].empty()) {
        config.extra_params["sizes"].pop_back();
      }
    }

    if (!size_range.empty()) {
      config.extra_params["sizes_range"] = size_range;
    }

    if (!args.benchmarks_.empty()) {
      config.benchmark_filter = "";
      for (const auto& b : args.benchmarks_) {
        config.benchmark_filter += b + ",";
      }
      config.benchmark_filter.pop_back();
    }
  }

  return args;
}

std::string Args::to_string() const {
  std::string str;

  switch (command_) {
    case Command::Info: str = "info"; break;
    case Command::List: str = "list"; break;
    case Command::Run: str = "run"; break;
    case Command::Selftest: str = "selftest"; break;
    case Command::Help: str = "help"; break;
  }

  return str;
}

}
