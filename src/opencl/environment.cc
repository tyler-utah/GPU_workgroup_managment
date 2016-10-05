#include "opencl/environment.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "opencl/opencl.h"

namespace opencl {

Execution& Execution::SetGlobalSize(const cl::NDRange& global_size) {
  global_size_ = global_size;
  return *this;
}

Execution& Execution::SetLocalSize(const cl::NDRange& local_size) {
  local_size_ = local_size;
  return *this;
}

cl::Event Execution::Enqueue(cl::CommandQueue command_queue) {
  cl::Event event;
  cl_int err = command_queue.enqueueNDRangeKernel(kernel_, cl::NDRange(),
                                                  global_size_, local_size_,
                                                  nullptr, &event);
  CHECK_OPENCL(err, "Enqueue kernel.");
  return event;
}

Context Context::CreateContextFromPlatform(const cl::Platform platform) {
  return CreateContextFromPlatformDevices(platform, {});//TODO
}

Context Context::CreateContextFromPlatformDevices(
      const cl::Platform platform, const std::vector<cl::Device>& devices) {
  cl_context_properties properties[] =
      {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform()), 0};
  cl_int err;
  // Callback not provided, may miss extra info provided to the callback.
  cl::Context context(devices, properties, nullptr, nullptr, &err);
  CHECK_OPENCL(err, "Creating context.");
  return Context(context, platform, devices);
}

cl::CommandQueue Context::GetOrCreateCommandQueue(const std::string& id) {
  cl_int err;
  auto pair = command_queues_.emplace(
      id, cl::CommandQueue(context_, 0, &err));
  CHECK_OPENCL(err, "Creating command queue.");
  return pair.first->second;
}

cl::CommandQueue Context::GetOrCreateCommandQueue() {
  return GetOrCreateCommandQueue("");
}

cl::CommandQueue *Context::GetCommandQueue(const std::string& id) {
  auto it = command_queues_.find(id);
  return it == command_queues_.end() ? nullptr : &it->second;
}

void Context::RegisterCommandQueue(const std::string& id,
                                   const cl::CommandQueue command_queue) {
  command_queues_[id] = command_queue;
}

cl::Program Context::CreateProgramFromFile(
    const std::string& path, const std::string& options) {
  CreateProgramFromFileForDevices(path, options, devices_);
}

cl::Program Context::CreateProgramFromFileForDevices(
    const std::string& path, const std::string& options,
    const std::vector<cl::Device>& devices) {
  std::ifstream f(path);
  std::ostringstream oss;
  oss << f.rdbuf();
  CreateProgramFromStringForDevices(oss.str(), options, devices);
}

cl::Program Context::CreateProgramFromString(
    const std::string& source, const std::string& options) {
  return CreateProgramFromStringForDevices(source, options, devices_);
}

cl::Program Context::CreateProgramFromStringForDevices(
    const std::string& source, const std::string& options,
    const std::vector<cl::Device>& devices) {
  cl_int err;
  cl::Program program(context_, source, &err);
  CHECK_OPENCL(err, "Creating program.");
  err = program.build(devices_, options.c_str(), nullptr, nullptr);
  // Fetch build status and logs.
  for (const cl::Device device : devices) {
    cl_int err_info;
    cl_build_status status =
        program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device, &err_info);
    CHECK_OPENCL(err_info, "Getting program build status.");
    if (status != CL_BUILD_ERROR) {
      continue;
    }
    std::string dev_name = device.getInfo<CL_DEVICE_NAME>();
    std::string build_log =
        program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err_info);
    CHECK_OPENCL(err_info, "Getting program build log.");
    std::cerr << "Build log for device " << device() << " (" << dev_name << "):"
              << std::endl;
    std::cerr << build_log << std::endl;
  }
  CHECK_OPENCL(err, "Building program.");
  return program;
}

cl::Program *Context::GetProgram(const std::string& id) {
  auto it = programs_.find(id);
  return it == programs_.end() ? nullptr : &it->second;
}

void Context::RegisterProgram(const std::string& id,
                              const cl::Program program) {
  programs_[id] = program;
}

}  // namespace opencl
