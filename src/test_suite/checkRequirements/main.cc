/**
   This program checks the requirements for the mega kernel experiments,
   namely:

   - OpenCL 2.x platform

   - SVM capabilities: fine grained buffer, atomics
*/

/*---------------------------------------------------------------------------*/

#include <string>
#include <iostream>

#include "opencl/opencl.h"

/*---------------------------------------------------------------------------*/

void check_err(cl_int err, std::string msg) {
  if (err != CL_SUCCESS) {
    std::cout << "OpenCL call failed: " << msg << "\n";
    exit (EXIT_FAILURE);
  }
}

/*---------------------------------------------------------------------------*/

void check_device(cl::Device device, bool *valid_device) {
  cl_int err;
  cl::string s;
  cl_device_svm_capabilities svm_capabilities;
  bool svm_requirements;

  err = device.getInfo(CL_DEVICE_NAME, &s);
  check_err(err, "cl::Device.getInfo()");
  std::cout << "device: " << s << "\n";

  err = device.getInfo(CL_DEVICE_SVM_CAPABILITIES, &svm_capabilities);
  check_err(err, "cl::Device.getInfo()");

  svm_requirements = (svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
  std::cout << "svm fine grain buffer: " << svm_requirements << "\n";
  if (!svm_requirements) {
    std::cout << "INVALID device: does not provide SVM fine grain buffer capability\n";
    return;
  }

  svm_requirements = (svm_capabilities & CL_DEVICE_SVM_ATOMICS) != 0;
  std::cout << "svm atomics: " << svm_requirements << "\n";
  if (!svm_requirements) {
    std::cout << "INVALID device: does not provide SVM atomics capability\n";
    return;
  }

  *valid_device = true;
}

/*---------------------------------------------------------------------------*/

void check_platform(cl::Platform platform, bool *valid_device) {
  cl_int err;
  int i;
  cl::string s;
  size_t num_devices;

  err = platform.getInfo(CL_PLATFORM_NAME, &s);
  check_err(err, "cl::Platform.getInfo()");
  std::cout << "platform: " << s << "\n";

  err = platform.getInfo(CL_PLATFORM_VERSION, &s);
  check_err(err, "clGetPlatformInfo()");
  std::cout << "version: " << s << "\n";

  if (s.compare(0, 9, "OpenCL 2.") != 0) {
    std::cout << "INVALID platform: does not provide OpenCL 2.x\n";
    return;
  }

  // check devices
  std::vector<cl::Device> devices;
  err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (err == CL_DEVICE_NOT_FOUND) {
    num_devices = 0;
  } else {
    check_err(err, "cl::Platform.getDevices()");
    num_devices = devices.size();
  }
  std::cout << "num_devices: " << num_devices << "\n";

  for (i = 0; i < num_devices; i++) {
    std::cout << "\n";
    check_device(devices[i], valid_device);
  }
}

/*---------------------------------------------------------------------------*/

int main() {
  cl_int err;
  int i;
  size_t num_platforms;
  bool valid_device = false;

  std::cout << "Check requirements for mega-kernel experiments\n\n";

  // get platforms
  std::vector<cl::Platform> platforms;
  err = cl::Platform::get(&platforms);
  check_err(err, "cl::Platform::get()");
  num_platforms = platforms.size();
  std::cout << "num_platforms: " << num_platforms << "\n";

  // check platforms
  for (i = 0; i < num_platforms; i++) {
    std::cout << "\n";
    check_platform(platforms[i], &valid_device);
  }
  std::cout << "\n";

  // verdict
  if (valid_device) {
    std::cout << "SUCCESS: At least one device meet the requirements\n";
    return EXIT_SUCCESS;
  }

  std::cout << "FAILURE: No device found to meet the requirements\n";
  return EXIT_FAILURE;
}

/*---------------------------------------------------------------------------*/
