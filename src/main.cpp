#include "perfcli/cli/Args.hpp"
#include "perfcli/cli/Commands.hpp"
#include <iostream>

using namespace perfcli;

int main(int argc, char* argv[]) {
  try {
    Args args = Args::parse(argc, argv);

    switch (args.command()) {
      case Command::Info:
        return Commands::execute_info(args);
      case Command::List:
        return Commands::execute_list(args);
      case Command::Run:
        return Commands::execute_run(args);
      case Command::Help:
        return 0;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
