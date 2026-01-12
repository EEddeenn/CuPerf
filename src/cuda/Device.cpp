#include "cuperf/cuda/Device.hpp"
#include "cuperf/util/Error.hpp"
#include <sstream>
#include <iomanip>
#include <string_view>

namespace cuperf {

int GpuInfo::major_version() const {
  size_t dot_pos = compute_capability.find('.');
  if (dot_pos != std::string::npos) {
    return std::stoi(compute_capability.substr(0, dot_pos));
  }
  return 0;
}

int GpuInfo::minor_version() const {
  size_t dot_pos = compute_capability.find('.');
  if (dot_pos != std::string::npos && dot_pos + 1 < compute_capability.length()) {
    return std::stoi(compute_capability.substr(dot_pos + 1));
  }
  return 0;
}

bool GpuInfo::is_compute_capability_at_least(int major, int minor) const {
  int major_ver = major_version();
  int minor_ver = minor_version();
  if (major_ver > major) return true;
  if (major_ver < major) return false;
  return minor_ver >= minor;
}

bool GpuInfo::supports_warp_shuffle() const {
  return is_compute_capability_at_least(3, 0);
}

bool GpuInfo::supports_tensor_cores() const {
  return is_compute_capability_at_least(7, 0);
}

bool GpuInfo::supports_fp16_tensor_cores() const {
  return is_compute_capability_at_least(7, 0);
}

bool GpuInfo::supports_bf16_tensor_cores() const {
  return is_compute_capability_at_least(8, 0);
}

DeviceManager& DeviceManager::instance() {
  static DeviceManager instance;
  return instance;
}

DeviceManager::DeviceManager() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
}

int DeviceManager::device_count() const {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

std::vector<GpuInfo> DeviceManager::enumerate_devices() const {
  std::vector<GpuInfo> devices;
  int count = device_count();
  devices.reserve(count);

  for (int i = 0; i < count; ++i) {
    try {
      devices.push_back(get_device_info(i));
    } catch (const CudaError&) {
      continue;
    }
  }

  return devices;
}

GpuInfo DeviceManager::get_device_info(int device_index) const {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));

  GpuInfo info;
  info.device_index = device_index;
  info.name = prop.name;

  std::ostringstream ss;
  ss << prop.major << "." << prop.minor;
  info.compute_capability = ss.str();

  info.sm_count = prop.multiProcessorCount;
  info.total_memory_bytes = prop.totalGlobalMem;
  info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);

  char uuid_str[128];
  snprintf(uuid_str, sizeof(uuid_str),
           "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           static_cast<unsigned char>(prop.uuid.bytes[0]),
           static_cast<unsigned char>(prop.uuid.bytes[1]),
           static_cast<unsigned char>(prop.uuid.bytes[2]),
           static_cast<unsigned char>(prop.uuid.bytes[3]),
           static_cast<unsigned char>(prop.uuid.bytes[4]),
           static_cast<unsigned char>(prop.uuid.bytes[5]),
           static_cast<unsigned char>(prop.uuid.bytes[6]),
           static_cast<unsigned char>(prop.uuid.bytes[7]),
           static_cast<unsigned char>(prop.uuid.bytes[8]),
           static_cast<unsigned char>(prop.uuid.bytes[9]),
           static_cast<unsigned char>(prop.uuid.bytes[10]),
           static_cast<unsigned char>(prop.uuid.bytes[11]),
           static_cast<unsigned char>(prop.uuid.bytes[12]),
           static_cast<unsigned char>(prop.uuid.bytes[13]),
           static_cast<unsigned char>(prop.uuid.bytes[14]),
           static_cast<unsigned char>(prop.uuid.bytes[15]));
  info.uuid = uuid_str;

  info.pci_bus_id = prop.pciBusID;
  info.pci_device_id = prop.pciDeviceID;

  return info;
}

void DeviceManager::set_device(int device_index) const {
  CUDA_CHECK(cudaSetDevice(device_index));
}

std::optional<int> DeviceManager::find_device_by_index(int index) const {
  int count = device_count();
  if (index >= 0 && index < count) {
    return index;
  }
  return std::nullopt;
}

std::optional<int> DeviceManager::find_device_by_uuid(const std::string& uuid) const {
  const std::string_view uuid_view(uuid);
  auto devices = enumerate_devices();
  for (const auto& device : devices) {
    if (std::string_view(device.uuid) == uuid_view) {
      return device.device_index;
    }
  }
  return std::nullopt;
}

}
