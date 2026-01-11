#include "perfcli/cuda/Device.hpp"
#include "perfcli/util/Error.hpp"
#include <sstream>
#include <iomanip>

namespace perfcli {

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
           "%08x-%04x-%04x-%04x-%04x%08x",
           (unsigned int)(prop.uuid.bytes[0] << 24 | prop.uuid.bytes[1] << 16 |
                          prop.uuid.bytes[2] << 8 | prop.uuid.bytes[3]),
           (unsigned int)(prop.uuid.bytes[4] << 8 | prop.uuid.bytes[5]),
           (unsigned int)(prop.uuid.bytes[6] << 8 | prop.uuid.bytes[7]),
           (unsigned int)(prop.uuid.bytes[8] << 8 | prop.uuid.bytes[9]),
           (unsigned int)(prop.uuid.bytes[8] << 8 | prop.uuid.bytes[9]),
           (unsigned int)(prop.uuid.bytes[10] << 24 | prop.uuid.bytes[11] << 16 |
                          prop.uuid.bytes[12] << 8 | prop.uuid.bytes[13]));
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
  auto devices = enumerate_devices();
  for (const auto& device : devices) {
    if (device.uuid == uuid) {
      return device.device_index;
    }
  }
  return std::nullopt;
}

}
