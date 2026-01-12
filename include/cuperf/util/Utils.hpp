#pragma once

#include <string>
#include <cstddef>

namespace cuperf {

[[nodiscard]] size_t parse_size(const std::string& s);

[[nodiscard]] std::string format_size(size_t bytes);

}
