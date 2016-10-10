// Interface for interacting with parts of the OpenCL interface.

#ifndef OPENCL_INTERFACE_H_
#define OPENCL_INTERFACE_H_

#include <cstdlib>
#include <iostream>
#include <string>

#include "base/commandlineflags.h"
#include "opencl/opencl.h"

#define OPENCL_LIST DECLARE_bool(opencl_list); \
        volatile bool unused = FLAGS_opencl_list;

// List all of the platforms and devices available on this machine, and then
// immediately exit.
//DECLARE_bool(opencl_list);

namespace opencl {

// Get the platform for the specified name, or die trying.
// The name will match if the passed name is a substring of the platform name as
// given by getPlatformInfo().
// The first platform with a matching name will be selected, hence giving an
// ambiguous name will give an implementation-defined result.
cl::Platform GetPlatformOrDie(const std::string& platform_name);

// Works in the same way as GetPlatformOrDie().
// Will only search through devices available for the specified platform.
cl::Device GetDeviceOrDie(const cl::Platform platform,
                          const std::string& device_name);

// Check OpenCL error code, and die if it is not CL_SUCCESS.
#define CHECK_OPENCL(err, msg)                             \
  {                                                        \
    cl_int err_ = (err);                                   \
    if (err_ != CL_SUCCESS) {                              \
      std::cerr << "[" << __FILE__ << ":" << __LINE__      \
                << "] OpenCL CHECK FAIL with error code "  \
                << err_ << "(" << msg << ")" << std::endl; \
      exit(EXIT_FAILURE);                                  \
    }                                                      \
  }

}  // namespace opencl

#endif  // OPENCL_INTERFACE_H_
