#pragma once

#include <string>
#include <vector>
#include <map>
#include "cuperf/core/Types.hpp"

namespace cuperf {

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
