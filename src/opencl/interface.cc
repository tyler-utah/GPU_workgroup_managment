#include "opencl/interface.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "opencl/opencl.h"

namespace opencl {

cl::Platform GetPlatformOrDie(const std::string& platform_name) {
  std::vector<cl::Platform> platforms;
  cl_int err = cl::Platform::get(&platforms);
  CHECK_OPENCL(err, "Get platforms.");
  for (const cl::Platform platform : platforms) {
    std::string platform_name_ = platform.getInfo<CL_PLATFORM_NAME>(&err);
    CHECK_OPENCL(err, "Get platform name.");
    if (platform_name_.find(platform_name) != std::string::npos) {
      return platform;
    }
  }
  //TODO logging
  std::cerr << "Cannot find platform with name \"" << platform_name << "\""
            << std::endl;
  exit(EXIT_FAILURE);
}

cl::Device GetDeviceOrDie(const cl::Platform platform,
                          const std::string& device_name) {
  std::vector<cl::Device> devices;
  cl_int err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  CHECK_OPENCL(err, "Get devices.");
  for (const cl::Device device : devices) {
    std::string device_name_ = device.getInfo<CL_DEVICE_NAME>(&err);
    CHECK_OPENCL(err, "Get device name.");
    if (device_name_.find(device_name) != std::string::npos) {
      return device;
    }
  }
  //TODO logging
  std::cerr << "Cannot find device with name \"" << device_name << "\""
            << std::endl;
  exit(EXIT_FAILURE);
}

}  // namespace opencl
