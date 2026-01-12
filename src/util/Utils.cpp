#include "cuperf/util/Utils.hpp"
#include <sstream>
#include <iomanip>

namespace cuperf {

size_t parse_size(const std::string& s) {
  size_t value = std::stoull(s);
  if (!s.empty()) {
    char last = s.back();
    switch (last) {
      case 'K': case 'k': value *= 1024; break;
      case 'M': case 'm': value *= 1024 * 1024; break;
      case 'G': case 'g': value *= 1024 * 1024 * 1024; break;
    }
  }
  return value;
}

std::string format_size(size_t bytes) {
  const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
  int suffix_index = 0;
  double size = static_cast<double>(bytes);

  while (size >= 1024.0 && suffix_index < 4) {
    size /= 1024.0;
    suffix_index++;
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << size << suffixes[suffix_index];
  return oss.str();
}

}
