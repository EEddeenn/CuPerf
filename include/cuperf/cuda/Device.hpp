#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>

namespace cuperf {

struct GpuInfo {
  int device_index;
  std::string name;
  std::string compute_capability;
  int sm_count;
  size_t total_memory_mb;
  size_t total_memory_bytes;
  std::string uuid;
  int pci_bus_id;
  int pci_device_id;

  [[nodiscard]] int major_version() const;
  [[nodiscard]] int minor_version() const;

  [[nodiscard]] bool supports_warp_shuffle() const;
  [[nodiscard]] bool supports_tensor_cores() const;
  [[nodiscard]] bool supports_fp16_tensor_cores() const;
  [[nodiscard]] bool supports_bf16_tensor_cores() const;
  [[nodiscard]] bool is_compute_capability_at_least(int major, int minor) const;
};

class DeviceManager {
public:
  static DeviceManager& instance();

  [[nodiscard]] int device_count() const;
  [[nodiscard]] std::vector<GpuInfo> enumerate_devices() const;
  [[nodiscard]] GpuInfo get_device_info(int device_index) const;
  void set_device(int device_index) const;

  [[nodiscard]] std::optional<int> find_device_by_index(int index) const;
  [[nodiscard]] std::optional<int> find_device_by_uuid(const std::string& uuid) const;

private:
  DeviceManager();
  ~DeviceManager() = default;

  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;
};

}
