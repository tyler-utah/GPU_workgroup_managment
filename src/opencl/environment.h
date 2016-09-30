// Classes and functions for interacting with the OpenCL environment.
//
// The OpenCL environment contains a few components as follws:
// - Device: A physical device capable of running OpenCL code.
// - Platform: An OpenCL installation. Recognises various devices.
// - CommandQueue: Queues up operations for specific devices in a context.
// - Program: An OpenCL program that may or may not yet be compiled for specific
//            devices in a context.
// - Context: For a platform and set of devices, holds verious command queues
//            and programs.
// From this, we get the following set of relations:
// - Platform --->* Device
// - CommandQueue --->* Device
// - Program --->* Device
// - Context --->1 Platform, --->* Device, --->* CommandQueue, --->* Program
// Important notes:
// - The set of devices for a command queue and program must be a subset of the
//   devices for the context they belong to, therefore there is an implied
//   relation from CommandQueue and Program to a single Context.
//
// Kernels represent a runnable instance of a program. They will have their args
// set before being enqueued for execution on the devices the program was built
// for. They are reusable.
// - Program --->* Kernel
//
// The environment is augment with two classes: Context and Execution.
// The Context class encapsulates the OpenCL context, providing access to the
// same objects across different parts of the codebase, as well as simplifying
// part of the interface.
// The Environment class makes working with Kernel objects simpler, in
// particular, adding arguments, setting properties and enqueueing.
//
// Example 1: Running a single kernel on a platform and device:
//
//   Context context = CreateContextFromPlatformDevices(
//       GetPlatformOrDie("NAMD"), {GetDeviceOrDie("GPU")});
//   Execution(context.CreateProgramFromFile("~/adder.cl", "-cl-opt-disable"))
//       .SetGlobalSize(8, 8)
//       .SetLocalSize(2, 2)
//       .SetArgs(my_buffer, 2, "/add")
//       .Enqueue(context.GetOrCreateCommandQueue());
//
// Example 2: SVM usage in custom queue:
//
//   Context context = CreateContextFromPlatformDevices(
//       GetPlatformOrDie("NAMD"), {GetDeviceOrDie("GPU")});
//   context.RegisterProgram(
//       "mega_kernel", context.CreateProgramFromFile("~/adder.cl", options));
//   std::atomic<int> *flag = context.SVMNew<std::atomic<int>>(0);
//   Execution(context.GetProgram("mega_kernel"))
//       .SetGlobalSize(1)
//       .SetLocalSize(1)
//       .SetArgs(SVMArg(flag), data)
//       .Enqueue(context.GetOrCreateCommandQueue("mega_queue");

#ifndef OPENCL_ENVIRONMENT_H_
#define OPENCL_ENVIRONMENT_H_

#include <map>
#include <set>
#include <string>
#include <vector>

#include "opencl/opencl.h"

namespace opencl {

// An Execution abstracts away an OpenCL kernel, giving you an object with which
// can be enqueued on a command queue and is reusable.
class Execution {
 public:
  explicit Execution(const cl::Program program) : program_(program) {}

  // Set the arguments for kernels represented by this execution
  template <typename... Args>
  cl_int SetArgs(Args... args) {
    return SetArgsIdx(0, args);
  }

  // TODO set kernel size and enqueue.

 private:
  // Base case for end of the argument pack.
  SetArgsIdx(cl_uint idx) { return CL_SUCCESS; }
  // Set argument for non-memory parameter.
  template <typename Arg, typename... Args>
  SetArgsIdx(cl_uint idx, Arg arg, Args... args) {
    if ((cl_int err = kernel_.setArg(idx, arg)) != CL_SUCCESS) {
      return err;
    }
    return SetArgsIdx(idx + 1, args);
  }
  // Set argument for memory object. For now, must be cast to array.
  // TODO: Have simple mem object struct and create specialisation for that.
  template <typename Arg, size_t size, Args... args>
  SetArgsIdx(cl_uint, Arg arg[size], Args... args) {
    if ((cl_int err = kernel_.setArg(idx, size, arg)) != CL_SUCCESS) {
      return err;
    }
    return SetArgsIdx(idx + 1, args);
  }

  // The program for which the kernel is built from.
  cl::Program program_;
  // The kernel for this execution.
  cl::Kernel kernel_;
};

// Context
class Context {
 public:
  // Initialise context with all devices available for the platform.
  Context(const cl::Context context, const cl::Platform platform,
          const std::vector<cl::Device>& devices)
      : context_(context), platform_(platform),
        devices_(devices.begin(), devices.end()) {
  }
  // All memory is handled by the platform, so destructor does nothing.
  ~Context() {}
  // Although this does not own the underlying object, we must ensure the
  // information here is consistent. Accessing the underlying context through a
  // different Context, that refers to the same underlying context will cause
  // unforseen changes.
  Context(const Context& other) = delete;
  Context operator=(const Context& other) = delete;
  // Contexts can still be moved however.
  context(context&& other) = default;
  Context operator=(Context&& other) = default;

  // Construct a context using the specified platform and all available devices
  // for the platform.
  static Context CreateContextFromPlatform(const cl::Platform platform);
  // Construct a context using the specified platform and devices.
  static Context CreateContextFromPlatformDevices(
      const cl::Platform platform, const std::vector<cl::Device>& devices);

  // Gets the CommandQueue that id maps to, or creates it if it does not exist.
  cl::CommandQueue GetOrCreateCommandQueue(const std::string& id);
  // Gets or creates the context-wide CommandQueue. This is equivalent to
  // calling with the empty string "".
  cl::CommandQueue GetOrCreateCommandQueue();
  // Gets the CommandQueue that id maps to, or nullptr if it does not exist.
  cl::CommandQueue *GetCommandQueue(const std::string& id) const;
  // Register a custom created CommandQueue. It is down to the user to ensure
  // CommandQueue was created with the appropriate context.
  void RegisterCommandQueue(const std::string& id,
                            const cl::CommandQueue command_queue);

  // Create, compile and link a program either from file or source. If no
  // devices are specified, all of the available devices are used.
  cl::Program CreateProgramFromFile(
      const std::string& path, const std::string& options);
  cl::Program CreateProgramFromFileForDevices(
      const std::string& path, const std::string& options,
      const std::vector<cl::Device>& devices);
  cl::Program CreateProgramFromString(
      const std::string& source, const std::string& options);
  cl::Program CreateProgramFromStringForDevices(
      const std::string& source, const std::string& options,
      const std::vector<cl::Device>& devices);
  // Gets the program that id maps to, or nullptr if it does not exist.
  cl::Program *GetProgram(const std::string& id) const;
  // Register an already created Program. It is down to the user to ensure
  // Program was created with the appropriate context.
  void RegisterProgram(const std::string& id, const cl::Program program);

  // SVM Allocation.
  // Example usage: std::atomic<int> *a = context.SVMNew<std::atomic<int>>(0)
  template <typename T, typename... Args>
  T *SVMNew(Args&&... args) {
    static cl_svm_mem_flags default_flags =
        CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS;
    void *mem = clSVMAlloc(context_, default_flags, sizeof(T), 0);
    return new(mem) T(args);
  }
  template <typename T>
  void SVMDelete(T *t) {
    t->~T();
    clSVMFree(context_, t);
  }

  // Get the underlying OpenCL reference classes used by this context.
  cl::Context GetContext() const;
  cl::Platform GetPLatform() const;
  const std::vector<cl::Device>& GetDevices() const;

 private:
  // Raw info for each context.

  // The underlying context.
  cl::Context context_;
  // Platform and devices for which this context is interacting with.
  cl::Platform platform_;
  std::set<cl::Device> devices_;
  // Active command queues, mapped by a string identifier. No identifier is the
  // context wide queue, for which all command queues are active.
  std::map<std::string, cl::CommandQueue> command_queues_;
  // Active programs, mapped by a string identifier.
  std::map<std::string, cl::Program> programs_;

  // The raw info contains the OpenCL handlers to an underlying instantiation,
  // for which the details are not available. We define some information objects
  // to store some useful info.

  // TODO these are prob not necessary.
  // Command queue info.
  struct CommandQueueInfo {
    std::set<cl::Device> devices_;
  };
  std::map<cl::CommandQueue, CommandQueueInfo> command_queue_info_;

  // Program info.
  struct ProgramInfo {
    std::set<cl::Device> devices_;
  };
  std::map<cl::Program, ProgramInfo> program_info_;
};

}  // namespace opencl

#endif  // OPENCL_ENVIRONMENT_H_
