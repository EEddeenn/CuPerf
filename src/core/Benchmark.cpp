#include "perfcli/core/Benchmark.hpp"

namespace perfcli {

BenchmarkContext::BenchmarkContext(int device, int stream_count)
    : device_index(device), streams(create_streams(stream_count)) {}

void BenchmarkContext::sync_all() {
  for (const auto& stream : streams) {
    stream->sync();
  }
}

bool Benchmark::is_supported(const GpuInfo& gpu) const {
  return true;
}

BenchmarkRegistry& BenchmarkRegistry::instance() {
  static BenchmarkRegistry instance;
  return instance;
}

void BenchmarkRegistry::register_benchmark(const std::string& name, BenchmarkFactory factory) {
  factories_[name] = std::move(factory);
}

std::unique_ptr<Benchmark> BenchmarkRegistry::create(const std::string& name) const {
  auto it = factories_.find(name);
  if (it != factories_.end()) {
    return it->second();
  }
  return nullptr;
}

std::vector<std::string> BenchmarkRegistry::list_benchmarks() const {
  std::vector<std::string> names;
  for (const auto& [name, _] : factories_) {
    names.push_back(name);
  }
  return names;
}

std::vector<BenchmarkSpec> BenchmarkRegistry::get_all_metadata() const {
  std::vector<BenchmarkSpec> specs;
  for (const auto& [name, factory] : factories_) {
    auto benchmark = factory();
    specs.push_back(benchmark->metadata());
  }
  return specs;
}

std::vector<std::string> BenchmarkRegistry::filter_by_tag(BenchmarkTag tag) const {
  std::vector<std::string> names;
  for (const auto& [name, factory] : factories_) {
    auto benchmark = factory();
    const auto& tags = benchmark->metadata().tags;
    if (std::find(tags.begin(), tags.end(), tag) != tags.end()) {
      names.push_back(name);
    }
  }
  return names;
}

bool BenchmarkRegistry::exists(const std::string& name) const {
  return factories_.find(name) != factories_.end();
}

BenchmarkRegistrar::BenchmarkRegistrar(const std::string& name, BenchmarkFactory factory) {
  BenchmarkRegistry::instance().register_benchmark(name, std::move(factory));
}

}
