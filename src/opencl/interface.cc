#include "opencl/interface.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "base/commandlineflags.h"
#include "opencl/opencl.h"

namespace opencl {
bool ListPlatformsAndDevices(const char *flag_name, bool value);
}  // namespace opencl

DEFINE_bool(opencl_list, false,
            "List all of the platforms and devices available on this machine, "
            "and then immediately exit.");
static const bool opencl_list_dummy = flags::RegisterFlagValidator(
    &FLAGS_opencl_list, &opencl::ListPlatformsAndDevices);

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

bool ListPlatformsAndDevices(const char *flag_name, bool value) {
  if (!value) {
    return true;
  }
  std::vector<cl::Platform> platforms;
  cl_int err = cl::Platform::get(&platforms);
  CHECK_OPENCL(err, "Get platforms.");
  for (const cl::Platform platform : platforms) {
    // Print info for this platform.
    std::cout << "PLT NAME: " << platform.getInfo<CL_PLATFORM_NAME>(&err)
              << std::endl;
    CHECK_OPENCL(err, "Get platform name.");
    std::cout << "PLT VENDOR: " << platform.getInfo<CL_PLATFORM_VENDOR>(&err)
              << std::endl;
    CHECK_OPENCL(err, "Get platform vendor.");
    std::cout << "PLT CL VERSION: "
              << platform.getInfo<CL_PLATFORM_VERSION>(&err)
              << std::endl;
    CHECK_OPENCL(err, "Get platform version.");
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    CHECK_OPENCL(err, "Get devices.");
    for (const cl::Device device : devices) {
      // Print info for this platform-device.
      std::cout << "  DEV NAME: " << device.getInfo<CL_DEVICE_NAME>(&err)
                << std::endl;
      CHECK_OPENCL(err, "Get device name.");
      std::cout << "  DEV VENDOR: " << device.getInfo<CL_DEVICE_VENDOR>(&err)
                << std::endl;
      CHECK_OPENCL(err, "Get device vendor.");
      std::cout << "  DEV CL VERSION: "
                << device.getInfo<CL_DEVICE_VERSION>(&err)
                << std::endl;
      CHECK_OPENCL(err, "Get device version.");
      std::cout << "  DEV TYPE: " << device.getInfo<CL_DEVICE_TYPE>(&err)
                << std::endl;
      CHECK_OPENCL(err, "Get device type.");
      // Sizes for this device
      std::cout << "  DEV MAX WG SIZE: "
                << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err)
                << std::endl;
      CHECK_OPENCL(err, "Get device max work group size.");
      cl_uint wi_dim = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(&err);
      CHECK_OPENCL(err, "Get device max work item dimension count.");
      std::vector<cl::size_type> wi_dims =
          device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>(&err);
      CHECK_OPENCL(err, "Get device max work item dimensions.");
      std::cout << "  DEV MAX WI DIMS COUNT: " << wi_dim << std::endl;
      std::cout << "  DEV MAX WI DIMS: [" << wi_dims[0];
      for (size_t dim = 1; dim < wi_dim; ++dim) {
        std::cout << ", " << wi_dims[dim];
      }
      std::cout << "]" << std::endl;
      std::cout << "  ========================================" << std::endl;
    }
    std::cout << "========================================" << std::endl;
  }
  exit(EXIT_SUCCESS);
  return true;
}

}  // namespace opencl
