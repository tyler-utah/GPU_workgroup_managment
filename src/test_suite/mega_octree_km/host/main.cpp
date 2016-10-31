#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include<iostream>
#include <vector>
#include <limits.h>

// Must be loaded early because it defines CL_XXX_TYPE
#include "scheduler_rt/rt_common/cl_types.h"

#include "base/commandlineflags.h"
#include "opencl/opencl.h"
#include "cl_execution.h"
#include "discovery.h"
#include "cl_communicator.h"
#include "../common/restoration_ctx.h"
#include "../common/octree.h"
#include "cl_scheduler.h"
#include "kernel_ctx.h"
#include "iw_barrier.h"
#include "base/file.h"

DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_bool(list, false, "List OpenCL platforms and devices");
DEFINE_string(scheduler_rt_path, "scheduler_rt/rt_device", "Path to scheduler runtime includes");
DEFINE_string(restoration_ctx_path, "test_suite/mega_octree_km/common/", "Path to restoration context");
DEFINE_int32(non_persistent_wgs, 2, "ratio of workgroups to send to non-persistent task. Special values are (-1) to send all but one workgroup and (-2) to send one workgroup");
DEFINE_int32(skip_tasks, 0, "flag to say if non persistent tasks should be skipped: 0 - don't skip, 1 - skip");

/*===========================================================================*/
// specific to octree

DEFINE_int32(numParticles, 100000, "number of particles to treat");
DEFINE_int32(maxChildren, 20, "maximum number of children");
DEFINE_int32(threads, 64, "number of threads");
DEFINE_int32(num_iterations, 300, "number of iterations");
static const unsigned int MAXTREESIZE = 11000000;

// see some octree types definitions in common/octree.h

/*===========================================================================*/

using namespace std;

const char *kernel_file = XSTR(KERNEL_FILE);

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

int get_app_kernels(CL_Execution &exec) {
  int ret = CL_SUCCESS;
  exec.exec_kernels["mega_kernel"] = cl::Kernel(exec.exec_program, "mega_kernel", &ret);
  return ret;
}

/* ------------------------------------------------------------------------- */

// genrand_real1() is defined in rand.cc
double genrand_real1(void);

/* ------------------------------------------------------------------------- */

int main(int argc, char *argv[]) {

  flags::ParseCommandLineFlags(&argc, &argv, true);

  CL_Execution exec;
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

  exec.exec_device = devices[FLAGS_platform_id][FLAGS_device_id];

  printf("Using GPU: %s\n", exec.getExecDeviceName().c_str());

  cl::Context context(exec.exec_device);
  exec.exec_context = context;
  cl::CommandQueue queue(exec.exec_context);
  exec.exec_queue = queue;

  // Should be built into the cmake file. Haven't thought of how to do this yet.
  err = exec.compile_kernel(kernel_file, file::Path(FLAGS_scheduler_rt_path), file::Path(FLAGS_restoration_ctx_path));

  check_ocl(err);

  get_app_kernels(exec);

  // set up the discovery protocol
  Discovery_ctx d_ctx;
  mk_init_discovery_ctx(&d_ctx);
  cl::Buffer d_ctx_mem (exec.exec_context, CL_MEM_READ_WRITE, sizeof(Discovery_ctx));
  err = exec.exec_queue.enqueueWriteBuffer(d_ctx_mem, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx);
  check_ocl(err);

  // scheduler context
  CL_Scheduler_ctx s_ctx;
  mk_init_scheduler_ctx(&exec, &s_ctx);

  IW_barrier h_bar;
  for (int i = 0; i < MAX_P_GROUPS; i++) {
    h_bar.barrier_flags[i] = 0;
  }
  h_bar.phase = 0;

  cl::Buffer d_bar(exec.exec_context, CL_MEM_READ_WRITE, sizeof(IW_barrier));
  err = exec.exec_queue.enqueueWriteBuffer(d_bar, CL_TRUE, 0, sizeof(IW_barrier), &h_bar);
  check_ocl(err);

  // kernel contexts for the graphics kernel and persistent kernel
  cl::Buffer d_graphics_kernel_ctx(exec.exec_context, CL_MEM_READ_WRITE, sizeof(Kernel_ctx));
  cl::Buffer d_persistent_kernel_ctx(exec.exec_context, CL_MEM_READ_WRITE, sizeof(Kernel_ctx));

  // set dummy args to enable discovery protoclol
  int arg_index = 0;
  int total_args_for_both_tasks = 3 /* graphics */ + 17 /* persistent */;
  for (arg_index = 0; arg_index < total_args_for_both_tasks; arg_index++) {
    err = exec.exec_kernels["mega_kernel"].setArg(arg_index, NULL);
    check_ocl(err);
  }
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_bar);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_ctx_mem);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_graphics_kernel_ctx);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_persistent_kernel_ctx);
  arg_index++;
  check_ocl(err);
  err = set_scheduler_args(&exec.exec_kernels["mega_kernel"], &s_ctx, arg_index);
  check_ocl(err);

  // Set up the communicator
  CL_Communicator cl_comm(exec, "mega_kernel", s_ctx, &d_ctx_mem);

  int occupancy_bound = cl_comm.get_occupancy_bound(FLAGS_threads);
  int max_workgroups = occupancy_bound - 1;
  printf("occupancy_bound %d\n", occupancy_bound);
  fflush(stdout);
  cl::NDRange global_size(occupancy_bound * FLAGS_threads);
  cl::NDRange local_size(FLAGS_threads);

  // Reduce kernel args.
  int graphics_arr_length = 1048576;
  cl_int * h_graphics_buffer = (cl_int *) malloc(sizeof(cl_int) * graphics_arr_length);
  int arr_min = INT_MAX;
  for (int i = 0; i < graphics_arr_length; i++) {
    int loop_int = rand() + 1;
    if (loop_int < arr_min) {
      arr_min = loop_int;
    }
    h_graphics_buffer[i] = loop_int;
  }

  cl::Buffer d_graphics_buffer(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int) * graphics_arr_length);
  err = exec.exec_queue.enqueueWriteBuffer(d_graphics_buffer, CL_TRUE, 0, sizeof(cl_int) * graphics_arr_length, h_graphics_buffer);

  cl_int * graphics_result = (cl_int*) clSVMAlloc(exec.exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
  *graphics_result = INT_MAX;


  // ----------------------------------------------------------------------
  // persistent kernel args

  // The number of pools for work-stealing is bounded by the max
  // possible number of workgroups, i.e., the value in
  // 'participating_groups'.

  int num_pools = max_workgroups;

  cout << "==== octree persistent kernel args ======" << endl;
  cout << "  numParticles: " << FLAGS_numParticles << endl;
  cout << "  threads: " << FLAGS_threads << endl;
  cout << "  num_pools: " << num_pools << endl;
  cout << "  maxChildren: " << FLAGS_maxChildren << endl;
  cout << "  num_iterations: " << FLAGS_num_iterations << endl;
  cout << "===================" << endl;

  // Hugues: this 'maxlength' value is also hardcoded in CUDA version,
  // see the 'dequeuelength' variable in CUDA
  unsigned int maxlength = 256;
  cl::Buffer d_num_iterations(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
  cl::Buffer randdata(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int) * 128);
  cl::Buffer maxl(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
  cl::Buffer particles(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_float4) * FLAGS_numParticles);
  cl::Buffer newParticles(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_float4) * FLAGS_numParticles);
  cl::Buffer tree(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint)*MAXTREESIZE);
  cl::Buffer treeSize(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint));
  cl::Buffer particlesDone(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint));
  cl::Buffer stealAttempts(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint));
  cl::Buffer deq(exec.exec_context, CL_MEM_READ_WRITE, sizeof(Task) * maxlength * num_pools);
  cl::Buffer dh(exec.exec_context, CL_MEM_READ_WRITE, sizeof(DequeHeader) * num_pools);

  cl_int num_iterations = FLAGS_num_iterations;
  err = exec.exec_queue.enqueueWriteBuffer(d_num_iterations, CL_TRUE, 0, sizeof(cl_int), &(num_iterations));
  check_ocl(err);
  err = exec.exec_queue.enqueueFillBuffer(tree, 0, 0, sizeof(cl_uint)*MAXTREESIZE);
  check_ocl(err);
  err = exec.exec_queue.enqueueFillBuffer(deq, 0, 0, sizeof(Task) * maxlength * num_pools);
  check_ocl(err);
  err = exec.exec_queue.enqueueFillBuffer(dh, 0, 0, sizeof(DequeHeader) * num_pools);
  check_ocl(err);

  // ----------------------------------------------------------------------
  // generate particles
  {
    cl_float4* lparticles = new cl_float4[FLAGS_numParticles];

    char fname[256];
    snprintf(fname, 256, "octreecacheddata-%dparticles.dat", FLAGS_numParticles);
    FILE* f = fopen(fname, "rb");
    if (!f) {
      cout << "Generating and caching data" << endl;

      int clustersize = 100;
      for (int i = 0; i < (FLAGS_numParticles / clustersize); i++) {
        float x = ((float)genrand_real1()*800.0f-400.0f);
        float y = ((float)genrand_real1()*800.0f-400.0f);
        float z = ((float)genrand_real1()*800.0f-400.0f);

        for (int x = 0; x < clustersize; x++) {
          lparticles[i*clustersize+x].s[0] = x + ((float)genrand_real1()*100.0f-50.0f);
          lparticles[i*clustersize+x].s[1] = y + ((float)genrand_real1()*100.0f-50.0f);
          lparticles[i*clustersize+x].s[2] = z + ((float)genrand_real1()*100.0f-50.0f);
        }
      }

      FILE* f = fopen(fname,"wb");
      fwrite(lparticles,sizeof(cl_float4), FLAGS_numParticles,f);
      fclose(f);
    } else {
      cout << "Read particle data from a file" << endl;
      fread(lparticles,sizeof(cl_float4), FLAGS_numParticles,f);
      fclose(f);
    }

    exec.exec_queue.enqueueWriteBuffer(particles, CL_TRUE, 0, sizeof(cl_float4) * FLAGS_numParticles, lparticles);
    delete lparticles;
  }

  cout << "Done with generating particles data" << endl;

  // ----------------------------------------------------------------------

  // Setting the args
  arg_index = 0;

  // // Set the args for graphics kernel
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, graphics_arr_length);
  arg_index++;
  err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, d_graphics_buffer);
  arg_index++;
  err |= clSetKernelArgSVMPointer(exec.exec_kernels["mega_kernel"](), arg_index, graphics_result);
  arg_index++;
  check_ocl(err);

  // Set args for persistent kernel
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_num_iterations);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, randdata);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, maxl);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, particles);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, newParticles);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, tree);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, FLAGS_numParticles);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, treeSize);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, particlesDone);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, FLAGS_maxChildren);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, stealAttempts);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, num_pools);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, deq);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, dh);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, maxlength);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, NULL);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, NULL);
  arg_index++;
  check_ocl(err);

  // Set barrier arg
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_bar);
  arg_index++;
  check_ocl(err);

  // Set arg for discovery protocol
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_ctx_mem);
  arg_index++;
  check_ocl(err);

  // Set arg for kernel contexts
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_graphics_kernel_ctx);
  arg_index++;
  check_ocl(err);
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_persistent_kernel_ctx);
  arg_index++;
  check_ocl(err);

  err = set_scheduler_args(&exec.exec_kernels["mega_kernel"], &s_ctx, arg_index);
  check_ocl(err);

  // Launch the mega kernel

  std::vector<time_stamp> response_time;
  std::vector<time_stamp> execution_time;
  int error = 0;

  int workgroups_for_non_persistent = 0;

  err = exec.exec_queue.flush();
  check_ocl(err);
  exec.exec_queue.finish();
  check_ocl(err);
  cout << "Launch mega kernel" << endl;
  err = cl_comm.launch_mega_kernel(global_size, local_size);
  check_ocl(err);

  if (FLAGS_non_persistent_wgs == -1) {
    workgroups_for_non_persistent = max_workgroups - 1;
  }
  else if (FLAGS_non_persistent_wgs == -2) {
    workgroups_for_non_persistent = 1;
  }
  else {
    workgroups_for_non_persistent = max_workgroups / FLAGS_non_persistent_wgs;
  }

  cout << "Send persistent task" << endl;
  cl_comm.send_persistent_task(num_pools);
  cl_comm.my_sleep(50);

  while (cl_comm.is_executing_persistent() && !FLAGS_skip_tasks) {
    *graphics_result = INT_MAX;
    cout << "Send graphic task" << endl;
    time_ret timing_info = cl_comm.send_task_synchronous(workgroups_for_non_persistent, "first");
    response_time.push_back(timing_info.second);
    execution_time.push_back(timing_info.first);

    int g_result = *graphics_result;
    if (g_result != 1) {
      error = 1;
    }
    cl_comm.my_sleep(100);
  }

  cout << "Send quit signal" << endl;
  cl_comm.send_quit_signal();
  cout << "Wait for queue to finish" << endl;
  err = exec.exec_queue.finish();
  check_ocl(err);
  cout << "Mega kernel has terminated" << endl;

  // ---------- Hugues: collect results from persistent kernel octree ----------
  int maxMem;
  unsigned int* htree;
  unsigned int htreeSize;
  unsigned int hparticlesDone;
  unsigned int hstealAttempts;

  err = exec.exec_queue.enqueueReadBuffer(maxl, CL_TRUE, 0, sizeof(cl_int), &maxMem);
  check_ocl(err);
  err = exec.exec_queue.enqueueReadBuffer(particlesDone, CL_TRUE, 0, sizeof(cl_uint), &hparticlesDone);
  check_ocl(err);
  err = exec.exec_queue.enqueueReadBuffer(treeSize, CL_TRUE, 0, sizeof(cl_uint), &htreeSize);
  check_ocl(err);
  htree = new unsigned int[MAXTREESIZE];
  err = exec.exec_queue.enqueueReadBuffer(tree, CL_TRUE, 0, sizeof(cl_uint) * MAXTREESIZE, htree);
  check_ocl(err);
  err = exec.exec_queue.enqueueReadBuffer(stealAttempts, CL_TRUE, 0, sizeof(cl_uint), &hstealAttempts);
  check_ocl(err);
  err = exec.exec_queue.enqueueReadBuffer(d_num_iterations, CL_TRUE, 0, sizeof(cl_uint), &num_iterations);
  check_ocl(err);

  // ---------- print the stats
  unsigned int sum = 0;
  for(unsigned int i = 0; i < htreeSize; i++) {
    if (htree[i] & 0x80000000) {
      sum += htree[i] & 0x7fffffff;
    }
  }

  cout << "========== results for octree ==========" << endl;
  cout << "  Tree size: " << htreeSize << endl;
  cout << "  Particles in tree: " << sum << " (" << FLAGS_numParticles << ") [" << hparticlesDone << "]" << endl;
  cout << "  Steal attempts: " << hstealAttempts << endl;
  cout << "  num_iterations value after the run (should be 0): " << num_iterations << endl;
  cout << "====================" << endl;

  // ----------------- Hugues: octree: end of stats collecting ---------------

  cl_comm.print_summary();

  cout << "Generate stats files..." << endl;

  cl_comm.print_groups_time_data("tmp.txt");

  cl_comm.print_response_exec_data("tmp2.txt");

  cl_comm.print_response_and_execution_times("tmp3.txt");

  cl_comm.print_summary_file("tmp4.txt");

  free_scheduler_ctx(&exec, &s_ctx);

  return 0;
}
