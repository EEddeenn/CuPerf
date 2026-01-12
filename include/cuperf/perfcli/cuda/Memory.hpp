#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstddef>
#include "perfcli/util/Error.hpp"

namespace perfcli {

template<typename T>
class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count = 0) : data_(nullptr), size_(count * sizeof(T)) {
    if (size_ > 0) {
      CUDA_CHECK(cudaMalloc(&data_, size_));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() {
    reset();
  }

  void reset() {
    if (data_) {
      cudaFree(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  [[nodiscard]] T* get() const noexcept { return static_cast<T*>(data_); }
  [[nodiscard]] size_t size_bytes() const noexcept { return size_; }
  [[nodiscard]] size_t count() const noexcept { return size_ / sizeof(T); }
  [[nodiscard]] bool is_empty() const noexcept { return data_ == nullptr; }

  void memset(int value) {
    if (data_) {
      CUDA_CHECK(cudaMemset(data_, value, size_));
    }
  }

  void memset_async(int value, cudaStream_t stream) {
    if (data_) {
      CUDA_CHECK(cudaMemsetAsync(data_, value, size_, stream));
    }
  }

private:
  void* data_;
  size_t size_;
};

enum class HostMemoryType {
  Pageable,
  Pinned
};

template<typename T>
class HostBuffer {
public:
  explicit HostBuffer(size_t count = 0, HostMemoryType type = HostMemoryType::Pageable)
      : data_(nullptr), size_(count * sizeof(T)), type_(type) {
    if (size_ > 0) {
      if (type_ == HostMemoryType::Pinned) {
        CUDA_CHECK(cudaMallocHost(&data_, size_));
      } else {
        data_ = new T[count];
      }
    }
  }

  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;

  HostBuffer(HostBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_), type_(other.type_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  HostBuffer& operator=(HostBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = other.data_;
      size_ = other.size_;
      type_ = other.type_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~HostBuffer() {
    reset();
  }

  void reset() {
    if (data_) {
      if (type_ == HostMemoryType::Pinned) {
        cudaFreeHost(data_);
      } else {
        delete[] static_cast<T*>(data_);
      }
      data_ = nullptr;
      size_ = 0;
    }
  }

  [[nodiscard]] T* get() const noexcept { return static_cast<T*>(data_); }
  [[nodiscard]] size_t size_bytes() const noexcept { return size_; }
  [[nodiscard]] size_t count() const noexcept { return size_ / sizeof(T); }
  [[nodiscard]] bool is_empty() const noexcept { return data_ == nullptr; }
  [[nodiscard]] HostMemoryType type() const noexcept { return type_; }

private:
  void* data_;
  size_t size_;
  HostMemoryType type_;
};

}
