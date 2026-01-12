#include "cuperf/core/Benchmark.hpp"
#include "cuperf/benchmarks/MemcpyBandwidth.hpp"
#include "cuperf/benchmarks/KernelLaunchOverhead.hpp"
#include "cuperf/benchmarks/DeviceMemBandwidth.hpp"
#include "cuperf/benchmarks/ComputeThroughput.hpp"
#include "cuperf/benchmarks/Reduction.hpp"
#include "cuperf/benchmarks/TensorCore.hpp"

using namespace cuperf;

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
