#pragma once

#include <stdexcept>
#include <cuda_runtime.h>
#include <string>
#include <format>
#include <sstream>
#include <fmt/core.h>

namespace cuperf {

class CudaError : public std::runtime_error {
public:
  CudaError(cudaError_t error, const std::string& file, int line)
      : std::runtime_error(fmt::format("CUDA error at {}:{}: {} ({})",
                                        file, line,
                                        cudaGetErrorString(error),
                                        static_cast<int>(error))),
        error_(error) {}

  cudaError_t error_code() const noexcept { return error_; }

private:
  cudaError_t error_;
};

}

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      throw cuperf::CudaError(err, __FILE__, __LINE__); \
    } \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
