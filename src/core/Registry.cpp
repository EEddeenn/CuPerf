#include "perfcli/core/Benchmark.hpp"
#include "perfcli/benchmarks/MemcpyBandwidth.hpp"
#include "perfcli/benchmarks/KernelLaunchOverhead.hpp"
#include "perfcli/benchmarks/DeviceMemBandwidth.hpp"
#include "perfcli/benchmarks/ComputeThroughput.hpp"
#include "perfcli/benchmarks/Reduction.hpp"
#include "perfcli/benchmarks/TensorCore.hpp"

using namespace perfcli;

namespace {
  BenchmarkRegistrar reg_memcpy(
      "memcpy", []() { return std::make_unique<MemcpyBandwidth>(); });

  BenchmarkRegistrar reg_kernel_launch(
      "kernel_launch", []() { return std::make_unique<KernelLaunchOverhead>(); });

  BenchmarkRegistrar reg_device_mem(
      "device_mem", []() { return std::make_unique<DeviceMemBandwidth>(); });

  BenchmarkRegistrar reg_compute(
      "compute", []() { return std::make_unique<ComputeThroughput>(); });

  BenchmarkRegistrar reg_reduction(
      "reduction", []() { return std::make_unique<Reduction>(); });

  BenchmarkRegistrar reg_tensor_core(
      "tensor_core", []() { return std::make_unique<TensorCore>(); });
}
