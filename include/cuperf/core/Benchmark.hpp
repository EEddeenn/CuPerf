#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include "cuperf/core/Types.hpp"
#include "cuperf/cuda/Device.hpp"
#include "cuperf/cuda/Stream.hpp"

namespace cuperf {

class BenchmarkContext {
public:
  int device_index;
  std::vector<StreamPtr> streams;

  BenchmarkContext(int device, int stream_count);
  ~BenchmarkContext() = default;

  void sync_all();
};

class Benchmark {
public:
  virtual ~Benchmark() = default;

  virtual BenchmarkSpec metadata() const = 0;
  virtual bool is_supported(const GpuInfo& gpu) const;
  virtual void setup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) = 0;
  virtual void run_warmup(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) = 0;
  virtual BenchmarkResult run_measure(BenchmarkContext& ctx, const std::map<std::string, std::string>& params) = 0;
  virtual bool verify_result(BenchmarkContext& ctx);
  virtual void teardown(BenchmarkContext& ctx) = 0;
};

using BenchmarkFactory = std::function<std::unique_ptr<Benchmark>()>;

class BenchmarkRegistry {
public:
  static BenchmarkRegistry& instance();

  void register_benchmark(const std::string& name, BenchmarkFactory factory);

  std::unique_ptr<Benchmark> create(const std::string& name) const;
  std::vector<std::string> list_benchmarks() const;
  std::vector<BenchmarkSpec> get_all_metadata() const;
  std::vector<std::string> filter_by_tag(BenchmarkTag tag) const;

  bool exists(const std::string& name) const;

private:
  BenchmarkRegistry() = default;

  std::map<std::string, BenchmarkFactory> factories_;
};

class BenchmarkRegistrar {
public:
  BenchmarkRegistrar(const std::string& name, BenchmarkFactory factory);
};

}

// Manual registration - do not use this macro anymore
