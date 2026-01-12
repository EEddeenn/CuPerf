#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <map>
#include <sstream>
#include "cuperf/cuda/Device.hpp"

namespace cuperf {

enum class BenchmarkTag {
  Memory,
  Compute,
  Latency,
  MultiGpu
};

struct SystemInfo {
  std::string hostname;
  std::string os;
  std::string cpu_model;
  std::string cuda_runtime_version;
  std::string cuda_driver_version;
  std::vector<GpuInfo> gpus;

  std::string to_json() const;
};

enum class DataType {
  Float32,
  Float16,
  BFloat16,
  Int8,
  Int32,
  Float4
};

std::string data_type_to_string(DataType dtype);
DataType string_to_data_type(const std::string& str);
size_t data_type_size(DataType dtype);

enum class Direction {
  HostToDevice,
  DeviceToHost,
  DeviceToDevice
};

std::string direction_to_string(Direction dir);

struct RunConfig {
  int device_index = 0;
  int warmup_iterations = 50;
  int measured_iterations = 200;
  int sample_count = 30;
  int stream_count = 1;
  bool use_pinned_memory = false;
  bool use_async_copies = false;
  bool verify_results = false;
  bool enable_nvml = false;
  std::string output_json_file;
  std::string output_csv_file;
  std::string benchmark_filter;
  std::string tag_filter;

  std::map<std::string, std::string> extra_params;
};

struct BenchmarkSpec {
  std::string name;
  std::string description;
  std::vector<std::string> parameters;
  std::map<std::string, std::string> default_params;
  std::vector<BenchmarkTag> tags;
  std::vector<DataType> supported_types;

  std::string to_json() const;
};

struct BenchmarkResult {
  std::string benchmark_name;
  std::map<std::string, std::string> params;
  int device_index;

  std::vector<double> raw_samples_us;
  double median_us;
  double p95_us;
  double p99_us;
  double mean_us;
  double stddev_us;

  std::map<std::string, double> metrics;
  std::vector<std::string> warnings;

  bool success;

  std::string to_json(bool include_raw_samples = true) const;
};

}
