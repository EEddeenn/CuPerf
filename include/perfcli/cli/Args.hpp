#pragma once

#include <string>
#include <vector>
#include <map>
#include "perfcli/core/Types.hpp"

namespace perfcli {

enum class Command {
  Info,
  List,
  Run,
  Selftest,
  Help
};

class Args {
public:
  static Args parse(int argc, char* argv[]);

  Command command() const { return command_; }
  const RunConfig& config() const { return config_; }

  std::string to_string() const;

private:
  Args() = default;

  Command command_;
  RunConfig config_;
  std::vector<std::string> benchmarks_;
};

}
