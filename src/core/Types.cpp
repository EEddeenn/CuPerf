#include "cuperf/core/Types.hpp"
#include <fmt/core.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace cuperf {

std::string data_type_to_string(DataType dtype) {
  switch (dtype) {
    case DataType::Float32: return "fp32";
    case DataType::Float16: return "fp16";
    case DataType::BFloat16: return "bf16";
    case DataType::Int8: return "int8";
    case DataType::Int32: return "int32";
    case DataType::Float4: return "fp4";
  }
  return "unknown";
}

DataType string_to_data_type(const std::string& str) {
  if (str == "fp32" || str == "float32" || str == "float") return DataType::Float32;
  if (str == "fp16" || str == "float16" || str == "half") return DataType::Float16;
  if (str == "bf16" || str == "bfloat16") return DataType::BFloat16;
  if (str == "int8") return DataType::Int8;
  if (str == "int32" || str == "int") return DataType::Int32;
  if (str == "fp4" || str == "float4") return DataType::Float4;
  return DataType::Float32;
}

size_t data_type_size(DataType dtype) {
  switch (dtype) {
    case DataType::Float32: return sizeof(float);
    case DataType::Float16: return 2;
    case DataType::BFloat16: return 2;
    case DataType::Int8: return 1;
    case DataType::Int32: return sizeof(int32_t);
    case DataType::Float4: return 1;
  }
  return sizeof(float);
}

std::string direction_to_string(Direction dir) {
  switch (dir) {
    case Direction::HostToDevice: return "H2D";
    case Direction::DeviceToHost: return "D2H";
    case Direction::DeviceToDevice: return "D2D";
  }
  return "unknown";
}

std::string SystemInfo::to_json() const {
  std::string json = "{\n";
  json += "  \"hostname\": \"" + hostname + "\",\n";
  json += "  \"os\": \"" + os + "\",\n";
  json += "  \"cpu_model\": \"" + cpu_model + "\",\n";
  json += "  \"cuda_runtime_version\": \"" + cuda_runtime_version + "\",\n";
  json += "  \"cuda_driver_version\": \"" + cuda_driver_version + "\",\n";
  json += "  \"gpus\": [\n";

  for (size_t i = 0; i < gpus.size(); ++i) {
    const auto& gpu = gpus[i];
    json += "    {\n";
    json += "      \"device_index\": " + std::to_string(gpu.device_index) + ",\n";
    json += "      \"name\": \"" + gpu.name + "\",\n";
    json += "      \"compute_capability\": \"" + gpu.compute_capability + "\",\n";
    json += "      \"sm_count\": " + std::to_string(gpu.sm_count) + ",\n";
    json += "      \"total_memory_mb\": " + std::to_string(gpu.total_memory_mb) + ",\n";
    json += "      \"total_memory_bytes\": " + std::to_string(gpu.total_memory_bytes) + ",\n";
    json += "      \"uuid\": \"" + gpu.uuid + "\"\n";
    json += "    }";
    if (i < gpus.size() - 1) json += ",";
    json += "\n";
  }

  json += "  ]\n";
  json += "}";
  return json;
}

std::string BenchmarkSpec::to_json() const {
  std::string json = "{\n";
  json += "  \"name\": \"" + name + "\",\n";
  json += "  \"description\": \"" + description + "\",\n";
  json += "  \"parameters\": [";

  for (size_t i = 0; i < parameters.size(); ++i) {
    json += "\"" + parameters[i] + "\"";
    if (i < parameters.size() - 1) json += ", ";
  }

  json += "],\n";

  json += "  \"default_params\": {\n";
  bool first = true;
  for (const auto& [key, value] : default_params) {
    if (!first) json += ",\n";
    json += "    \"" + key + "\": \"" + value + "\"";
    first = false;
  }
  json += "\n  },\n";

  json += "  \"tags\": [";
  for (size_t i = 0; i < tags.size(); ++i) {
    switch (tags[i]) {
      case BenchmarkTag::Memory: json += "\"memory\""; break;
      case BenchmarkTag::Compute: json += "\"compute\""; break;
      case BenchmarkTag::Latency: json += "\"latency\""; break;
      case BenchmarkTag::MultiGpu: json += "\"multi-gpu\""; break;
    }
    if (i < tags.size() - 1) json += ", ";
  }
  json += "]\n";

  json += "}";
  return json;
}

std::string BenchmarkResult::to_json(bool include_raw_samples) const {
  std::string json = "{\n";
  json += "  \"benchmark_name\": \"" + benchmark_name + "\",\n";
  json += "  \"device_index\": " + std::to_string(device_index) + ",\n";
  json += "  \"success\": " + std::string(success ? "true" : "false") + ",\n";

  json += "  \"params\": {\n";
  bool first = true;
  for (const auto& [key, value] : params) {
    if (!first) json += ",\n";
    json += "    \"" + key + "\": \"" + value + "\"";
    first = false;
  }
  json += "\n  },\n";

  if (include_raw_samples && !raw_samples_us.empty()) {
    json += "  \"raw_samples_us\": [";
    for (size_t i = 0; i < raw_samples_us.size(); ++i) {
      json += fmt::format("{:.2f}", raw_samples_us[i]);
      if (i < raw_samples_us.size() - 1) json += ", ";
    }
    json += "],\n";
  }

  json += "  \"statistics\": {\n";
  json += "    \"median_us\": " + fmt::format("{:.2f}", median_us) + ",\n";
  json += "    \"mean_us\": " + fmt::format("{:.2f}", mean_us) + ",\n";
  json += "    \"stddev_us\": " + fmt::format("{:.2f}", stddev_us) + ",\n";
  json += "    \"p95_us\": " + fmt::format("{:.2f}", p95_us) + ",\n";
  json += "    \"p99_us\": " + fmt::format("{:.2f}", p99_us) + "\n";
  json += "  },\n";

  json += "  \"metrics\": {\n";
  first = true;
  for (const auto& [key, value] : metrics) {
    if (!first) json += ",\n";
    json += "    \"" + key + "\": " + fmt::format("{:.2f}", value);
    first = false;
  }
  json += "\n  }";

  if (!warnings.empty()) {
    json += ",\n  \"warnings\": [";
    for (size_t i = 0; i < warnings.size(); ++i) {
      json += "\"" + warnings[i] + "\"";
      if (i < warnings.size() - 1) json += ", ";
    }
    json += "]";
  }

  json += "\n}";
  return json;
}

}
