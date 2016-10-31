// This is an adaptation of skel.cpp

#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <limits.h>

#include "cl_types.h"
#include "base/commandlineflags.h"
#include "opencl/opencl.h"
#include "cl_execution.h"
#include "discovery.h"
#include "cl_communicator.h"
#include "restoration_ctx.h"
#include "cl_scheduler.h"
#include "kernel_ctx.h"
#include "iw_barrier.h"
#include "base/file.h"

DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_bool(list, false, "List OpenCL platforms and devices");
DEFINE_string(scheduler_rt_path, "scheduler_rt/rt_device", "Path to scheduler runtime includes");
DEFINE_string(output_timestamp_executing_groups, "timestamp_executing_groups", "Path to output timestamps and the number of persistent groups");
DEFINE_string(output_timestamp_non_persistent, "timestamp_non_persistent", "Path to output timestamps for non persistent tasks");
DEFINE_string(output_non_persistent_duration, "non_persistent_duration", "Path to output duration results for non persistent tasks");
DEFINE_string(output_summary, "summary", "output file for summary of results");
DEFINE_int32(non_persistent_wgs, 2, "ratio of workgroups to send to non-persistent task. Special values are (-1) to send all but one workgroup and (-2) to send one workgroup");
DEFINE_int32(skip_tasks, 0, "flag to say if non persistent tasks should be skipped: 0 - don't skip, 1 - skip");
DEFINE_int32(non_persistent_frequency, 100, "frequency in milliseconds to launch non_persistent tasks");
DEFINE_int32(is_AMD, 0, "flag to set on AMD chips to reduce occupancy as a work around to the defunct process issue");
DEFINE_int32(use_query_barrier, 0, "flag to use regular ckill barrier or query ckill barrer");

DEFINE_int32(run_non_persistent, 0, "Run only the non persistent task (solo run). If greater than 0, specifies the iterations");
DEFINE_int32(threads_per_wg, 256, "Threads per workgroup for non persistent task solo runs");
DEFINE_int32(num_wgs, 8, "Workgroups for non persistent task solo runs");
DEFINE_int32(merged_iterations, 1, "Iterations to run the merged kernel for");

DEFINE_int32(run_persistent, 0, "Run only the persistent task. If greater than 0, specifies the iterations");

/*---------------------------------------------------------------------------*/

DEFINE_string(restoration_ctx_path, "test_suite/os_freq/common/", "Path to restoration context");
DEFINE_string(merged_kernel_file, "test_suite/os_freq/device/merged.cl", "the path the mega kernel file");
DEFINE_string(persistent_kernel_file, "test_suite/os_freq/device/os_freq.cl", "the path the mega kernel file");
DEFINE_int32(num_iterations, 1000, "number of iterations");

using namespace std;

//From IWOCL tutorial (needs attribution)
unsigned getDeviceList(std::vector<std::vector<cl::Device> >& devices)
{
  // Get list of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // Enumerate devices
  for (unsigned int i = 0; i < platforms.size(); i++)
    {
      std::vector<cl::Device> plat_devices;
      platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
      devices.push_back(plat_devices);
      //devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
    }

  return devices.size();
}

//From IWOCL tutorial (needs attribution)
void list_devices() {

  std::vector<std::vector<cl::Device> > devices;
  getDeviceList(devices);

  // Print device names
  if (devices.size() == 0) {
    std::cout << "No devices found." << std::endl;
  }
  else {
    std::cout << std::endl;
    std::cout << "Platform,Devices:" << std::endl;
    for (unsigned j = 0; j < devices.size(); j++) {
      for (unsigned i = 0; i < devices[j].size(); i++) {
        std::cout << j << ", " << i << ": " << CL_Execution::getDeviceName(devices[j][i]) << std::endl;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

void reset_barrier(CL_Execution *exec, cl::Buffer d_bar) {

  IW_barrier h_bar;
  for (int i = 0; i < MAX_P_GROUPS; i++) {
    h_bar.barrier_flags[i] = 0;
  }
  h_bar.phase = 0;

  int err = exec->exec_queue.enqueueWriteBuffer(d_bar, CL_TRUE, 0, sizeof(IW_barrier), &h_bar);
  check_ocl(err);

}

/*---------------------------------------------------------------------------*/

void reset_discovery(CL_Execution *exec, cl::Buffer d_ctx_mem, bool get_occupancy) {

  Discovery_ctx d_ctx;
  if (get_occupancy) {
    mk_init_discovery_ctx_occupancy(&d_ctx);
  }
  else {
    mk_init_discovery_ctx(&d_ctx);
  }
  int err = exec->exec_queue.enqueueWriteBuffer(d_ctx_mem, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx);
  check_ocl(err);

}

/*---------------------------------------------------------------------------*/

int discovery_get_occupancy(CL_Execution *exec, cl::Buffer d_ctx_mem) {

  Discovery_ctx d_ctx;
  mk_init_discovery_ctx(&d_ctx);
  int err = exec->exec_queue.enqueueReadBuffer(d_ctx_mem, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx);
  check_ocl(err);
  return d_ctx.count;

}

/*---------------------------------------------------------------------------*/

int get_occupancy_d_ctx(CL_Execution *exec, cl::Kernel k, cl::Buffer d_ctx) {
	int err = exec->exec_queue.flush();
	check_ocl(err);
	err = exec->exec_queue.finish();
	check_ocl(err);
	err = exec->exec_queue.enqueueNDRangeKernel(k,
		cl::NullRange,
		cl::NDRange(FLAGS_threads_per_wg * MAX_P_GROUPS),
		cl::NDRange(FLAGS_threads_per_wg),
		NULL);
	check_ocl(err);
	err = exec->exec_queue.flush();
	check_ocl(err);
	err = exec->exec_queue.finish();
	check_ocl(err);

	return discovery_get_occupancy(exec, d_ctx);
}

/*---------------------------------------------------------------------------*/

int amd_check(int v) {
	if (FLAGS_is_AMD == 1) {
		v = (double)v * .75;
	}
	return v;
}

/*---------------------------------------------------------------------------*/

static uint64_t gettime_nanosec() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

/*---------------------------------------------------------------------------*/

int main(int argc, char *argv[]) {

  flags::ParseCommandLineFlags(&argc, &argv, true);

  int err = 0;

  if (FLAGS_list) {
    list_devices();
    exit(0);
  }

  std::vector<std::vector<cl::Device> > devices;
  getDeviceList(devices);
  if (FLAGS_platform_id < 0 || FLAGS_platform_id >= devices.size()) {
    printf("invalid platform id. Please use the --list option to view platforms and device ids\n");
    exit(0);
  }
  if (FLAGS_device_id < 0 || FLAGS_device_id >= devices[FLAGS_platform_id].size()) {
    printf("invalid device id. Please use the --list option to view platforms and device ids\n");
  }

  CL_Execution exec;

  exec.exec_device = devices[FLAGS_platform_id][FLAGS_device_id];
  printf("Using GPU: %s\n", exec.getExecDeviceName().c_str());

  cl::Context context(exec.exec_device);
  exec.exec_context = context;
  cl::CommandQueue queue(exec.exec_context, CL_QUEUE_PROFILING_ENABLE);
  exec.exec_queue = queue;



  // --------------------------------------------------0
  printf("OS FREQ\n");
  printf("Running with %d internal iterations\n", FLAGS_num_iterations);

  err = exec.compile_kernel(file::Path(FLAGS_persistent_kernel_file),
                                 file::Path(FLAGS_scheduler_rt_path),
                                 file::Path(FLAGS_restoration_ctx_path),
                                 FLAGS_use_query_barrier);
  check_ocl(err);
  exec.exec_kernels["persistent"] = cl::Kernel(exec.exec_program, "os_freq", &err);
  check_ocl(err);

  cl::Buffer d_num_iterations = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, FLAGS_num_iterations);
  err = exec.exec_queue.enqueueWriteBuffer(d_num_iterations, CL_TRUE, 0, sizeof(cl_int), &(FLAGS_num_iterations));
  check_ocl(err);

  int arg_index = 0;

  err = exec.exec_kernels["persistent"].setArg(arg_index, d_num_iterations);
  check_ocl(err);
  arg_index++;

  err = exec.exec_queue.flush();
  check_ocl(err);

  cl::Buffer d_bar(exec.exec_context, CL_MEM_READ_WRITE, sizeof(IW_barrier));
  reset_barrier(&exec, d_bar);
  err = exec.exec_kernels["persistent"].setArg(arg_index, d_bar);
  arg_index++;
  check_ocl(err);

  cl::Buffer d_ctx_mem(exec.exec_context, CL_MEM_READ_WRITE, sizeof(Discovery_ctx));
  reset_discovery(&exec, d_ctx_mem, true);
  err = exec.exec_kernels["persistent"].setArg(arg_index, d_ctx_mem);
  arg_index++;
  check_ocl(err);

  CL_Scheduler_ctx s_ctx;
  mk_init_scheduler_ctx(&exec, &s_ctx);
  err = set_scheduler_args(&(exec.exec_kernels["persistent"]), &s_ctx, arg_index);

  // int occupancy = get_occupancy_d_ctx(&exec, exec.exec_kernels["persistent"], d_ctx_mem);
  // int num_wgs = min(FLAGS_num_wgs, amd_check(occupancy));
  // printf("%d threads per workgroup, %d workgroups, %d occupancy, %d final size\n", FLAGS_threads_per_wg, FLAGS_num_wgs, occupancy, num_wgs);

  CL_Communicator::my_sleep(1000);

  int error = 0;
  int i = 0;

  err = exec.exec_queue.flush();
  check_ocl(err);
  err = exec.exec_queue.finish();
  check_ocl(err);

  // everything is in nanosec
  uint64_t start_loop = 0;
  uint64_t time_loop = 0;
  uint64_t time_diff = 0;
  cl::Event event;
  cl_ulong start_kernel = 0;
  cl_ulong end_kernel = 0;
  cl_ulong time_kernel = 0;

  while (true) {
    start_loop = gettime_nanosec();

	// print the stat of the previous loop, to be able to terminate
	// loop on a timing measurement
	time_diff = time_loop - time_kernel;
	printf("Loop   %10lld ns ", time_loop);
	printf("Kernel %10lld ns ", time_kernel);
	printf("Diff   %10lld ns\n", time_diff);

    reset_discovery(&exec, d_ctx_mem, false);
    reset_barrier(&exec, d_bar);
    restart_scheduler(&s_ctx);

    err = exec.exec_queue.enqueueNDRangeKernel(exec.exec_kernels["persistent"],
                                               cl::NullRange,
                                               cl::NDRange(FLAGS_threads_per_wg * MAX_P_GROUPS),
                                               cl::NDRange(FLAGS_threads_per_wg),
                                               NULL,
                                               &event);
    check_ocl(err);
    err = exec.exec_queue.flush();
    check_ocl(err);
    err = exec.exec_queue.finish();
    check_ocl(err);

    err = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_kernel);
    check_ocl(err);
    err = event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_kernel);
    check_ocl(err);

    time_kernel = end_kernel - start_kernel;
    time_loop = gettime_nanosec() - start_loop;
  }
}

/*---------------------------------------------------------------------------*/
