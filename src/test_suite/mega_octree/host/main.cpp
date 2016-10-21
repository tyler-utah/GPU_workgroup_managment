#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include<iostream>
#include <vector>
#include "limits.h"

// Should be in a directory somewhere probably. Or defined in CMake.
#define CL_INT_TYPE cl_int
#define ATOMIC_CL_INT_TYPE cl_int
#define CL_UCHAR_TYPE cl_uchar
#define MY_CL_GLOBAL

#include "base/commandlineflags.h"
#include "opencl/opencl.h"
#include "cl_execution.h"
#include "discovery.h"
#include "cl_communicator.h"
#include "../common/restoration_ctx.h"
#include "cl_scheduler.h"
#include "kernel_ctx.h"
#include "iw_barrier.h"
#include "base/file.h"

DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_bool(list, false, "List OpenCL platforms and devices");
DEFINE_string(scheduler_rt_path, "scheduler_rt/rt_device", "Path to scheduler runtime includes");
DEFINE_string(restoration_ctx_path, "test_suite/mega_octree/common/", "Path to restoration context");
//DEFINE_string(graph_file, "", "Path to the graph_file");
//DEFINE_string(output, "", "Path to output the result");
DEFINE_int32(non_persistent_wgs, 2, "ratio of workgroups to send to non-persistent task. Special values are (-1) to send all but one workgroup and (-2) to send one workgroup");
DEFINE_int32(skip_tasks, 0, "flag to say if non persistent tasks should be skipped: 0 - don't skip, 1 - skip");

/*===========================================================================*/
// specific to octree

DEFINE_int32(numParticles, 10000, "number of particles to treat");
DEFINE_int32(maxChildren, 50, "maximum number of children");
DEFINE_int32(blocks, 12, "number of blocks");
DEFINE_int32(threads, 32, "number of threads");
DEFINE_int32(num_iterations, 1, "number of iterations");
static const unsigned int MAXTREESIZE = 11000000;

/*---------------------------------------------------------------------------*/

typedef struct {
  cl_float4 middle;
  cl_bool flip;
  cl_uint end;
  cl_uint beg;
  cl_uint treepos;
} Task;

/*---------------------------------------------------------------------------*/

typedef struct {
  cl_int tail;
  cl_int head;
} DequeHeader;

/*---------------------------------------------------------------------------*/

typedef struct {
  Task *deq;
  DequeHeader* dh;
  unsigned int maxlength;
} DLBABP;

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

// get_num_participating_groups() launch the mega-kernel just to run the
// discovery protocol and obtain the number of participating
// groups.
int get_num_participating_groups(CL_Execution *exec)
{
  int err = 0;
  Discovery_ctx d_ctx;
  mk_init_discovery_ctx(&d_ctx);
  cl::Buffer d_ctx_mem (exec->exec_context, CL_MEM_READ_WRITE, sizeof(Discovery_ctx));
  err = exec->exec_queue.enqueueWriteBuffer(d_ctx_mem, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx);
  check_ocl(err);

  CL_Scheduler_ctx s_ctx;
  mk_init_scheduler_ctx(exec, &s_ctx);

  // Set up the communicator
  int local_size = FLAGS_threads;
  int wg_size = MAX_P_GROUPS;
  CL_Communicator cl_comm_disco(*exec, "mega_kernel", cl::NDRange(wg_size * local_size), cl::NDRange(local_size), s_ctx);

  IW_barrier h_bar;
  for (int i = 0; i < MAX_P_GROUPS; i++) {
    h_bar.barrier_flags[i] = 0;
  }
  h_bar.phase = 0;

  cl::Buffer d_bar(exec->exec_context, CL_MEM_READ_WRITE, sizeof(IW_barrier));
  err = exec->exec_queue.enqueueWriteBuffer(d_bar, CL_TRUE, 0, sizeof(IW_barrier), &h_bar);
  check_ocl(err);

  cl::Buffer d_graphics_kernel_ctx(exec->exec_context, CL_MEM_READ_WRITE, sizeof(Kernel_ctx));
  cl::Buffer d_persistent_kernel_ctx(exec->exec_context, CL_MEM_READ_WRITE, sizeof(Kernel_ctx));

  // Use a dummy buffer for the tasks args, which will not be started
  // anyway
  cl::Buffer dummy(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));

  int arg_index = 0;

  // dummy args for graphics
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, 0);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;

  // dummy args for octree
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, 0);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, 0);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, 0);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, dummy);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, 0);
  check_ocl(err);
  arg_index++;


  // relevant args for a run just to get the number of participating
  // workgroups
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, d_bar);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, d_ctx_mem);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, d_graphics_kernel_ctx);
  check_ocl(err);
  arg_index++;
  err = exec->exec_kernels["mega_kernel"].setArg(arg_index, d_persistent_kernel_ctx);
  check_ocl(err);
  arg_index++;
  err = set_scheduler_args(&(exec->exec_kernels["mega_kernel"]), &s_ctx, arg_index);
  check_ocl(err);

  // get number of participating workgroups
  err = exec->exec_queue.flush();
  check_ocl(err);
  err = exec->exec_queue.finish();
  check_ocl(err);
  err = cl_comm_disco.launch_mega_kernel();
  check_ocl(err);
  int participating_groups = cl_comm_disco.number_of_discovered_groups();
  cl_comm_disco.send_quit_signal();
  err = exec->exec_queue.finish();
  check_ocl(err);

  // free all buffers from the context
  // err = clReleaseMemObject(d_ctx_mem);
  // check_ocl(err);
  // err = clReleaseMemObject(d_bar);
  // check_ocl(err);
  // err = clReleaseMemObject(d_graphics_kernel_ctx);
  // check_ocl(err);
  // err = clReleaseMemObject(d_persistent_kernel_ctx);
  // check_ocl(err);
  // err = clReleaseMemObject(dummy);
  // check_ocl(err);
  // err = exec->exec_queue.finish();
  // check_ocl(err);

  return participating_groups;
}

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

  int participating_groups = get_num_participating_groups(&exec);

  cout << "number of participating workgroups: " << participating_groups << endl;

  // set up the discovery protocol
  Discovery_ctx d_ctx;
  mk_init_discovery_ctx(&d_ctx);
  cl::Buffer d_ctx_mem (exec.exec_context, CL_MEM_READ_WRITE, sizeof(Discovery_ctx));
  err = exec.exec_queue.enqueueWriteBuffer(d_ctx_mem, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx);
  check_ocl(err);

  // scheduler context
  CL_Scheduler_ctx s_ctx;
  mk_init_scheduler_ctx(&exec, &s_ctx);

  // Set up the communicator
  int local_size = FLAGS_threads;
  int wg_size = MAX_P_GROUPS;
  CL_Communicator cl_comm(exec, "mega_kernel", cl::NDRange(wg_size * local_size), cl::NDRange(local_size), s_ctx);

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

  int num_pools = participating_groups;

  cout << "==== persistent kernel args ======" << endl;
  cout << "  numParticles: " << FLAGS_numParticles << endl;
  cout << "  threads: " << FLAGS_threads << endl;
  cout << "  arg blocks: " << FLAGS_blocks << endl;
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

  IW_barrier octree_h_bar;
  for (int i = 0; i < MAX_P_GROUPS; i++) {
    octree_h_bar.barrier_flags[i] = 0;
  }
  octree_h_bar.phase = 0;
  // for sense reversal barrier
  octree_h_bar.counter = 0;
  octree_h_bar.sense = 0;

  cl::Buffer octree_d_bar(exec.exec_context, CL_MEM_READ_WRITE, sizeof(IW_barrier));
  err = exec.exec_queue.enqueueWriteBuffer(octree_d_bar, CL_TRUE, 0, sizeof(IW_barrier), &h_bar);
  check_ocl(err);

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
      printf("Generating and caching data\n");

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
      fread(lparticles,sizeof(cl_float4), FLAGS_numParticles,f);
      fclose(f);
    }

    exec.exec_queue.enqueueWriteBuffer(particles, CL_TRUE, 0, sizeof(cl_float4) * FLAGS_numParticles, lparticles);
    delete lparticles;
  }

  // ----------------------------------------------------------------------

  // Setting the args
  int arg_index = 0;

  // // Set the args for graphics kernel
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, graphics_arr_length);
  arg_index++;
  err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, d_graphics_buffer);
  arg_index++;
  err |= clSetKernelArgSVMPointer(exec.exec_kernels["mega_kernel"](), arg_index, graphics_result);
  arg_index++;
  check_ocl(err);

  // Set args for persistent kernel
  err = exec.exec_kernels["mega_kernel"].setArg(arg_index, octree_d_bar);
  arg_index++;
  check_ocl(err);
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
  err = cl_comm.launch_mega_kernel();
  check_ocl(err);

  if (FLAGS_non_persistent_wgs == -1) {
    workgroups_for_non_persistent = participating_groups - 1;
  }
  else if (FLAGS_non_persistent_wgs == -2) {
    workgroups_for_non_persistent = 1;
  }
  else {
    workgroups_for_non_persistent = participating_groups / FLAGS_non_persistent_wgs;
  }

  cout << "send persistent task with " << num_pools << " work groups" << endl;
  cl_comm.send_persistent_task(num_pools);

  while (cl_comm.is_executing_persistent() && !FLAGS_skip_tasks) {
    *graphics_result = INT_MAX;
    cout << " * start non-persistent task" << endl;
    time_ret timing_info = cl_comm.send_task_synchronous(workgroups_for_non_persistent, "first");
    response_time.push_back(timing_info.second);
    execution_time.push_back(timing_info.first);

    int g_result = *graphics_result;
    if (g_result != 1) {
      error = 1;
    }
    cl_comm.my_sleep(100);
  }

  cout << "send quit signal" << endl;
  cl_comm.send_quit_signal();
  cout << "ask queue to finish" << endl;
  err = exec.exec_queue.finish();
  check_ocl(err);

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

  cout << "number of participating groups: " << *(s_ctx.participating_groups) << endl;

  cout << "executed " << response_time.size() << " non-persistent tasks" << std::endl;

  for (int i = 0; i < response_time.size(); i++) {
    cout << "times " << i << ": " << cl_comm.nano_to_milli(response_time[i]) << " " << cl_comm.nano_to_milli(execution_time[i]) << endl;
  }

  cout << endl << "error: " << error << endl;

  cout << "persistent kernel time: " << cl_comm.nano_to_milli(cl_comm.get_persistent_time()) << " ms" << endl;

  cout << "non persistent kernels executed with: " << workgroups_for_non_persistent << " workgroups" << endl;

  cout << "total response time: " << cl_comm.reduce_times_ms(response_time) << " ms" << endl;

  cout << "average response time: " << cl_comm.get_average_time_ms(response_time) << " ms" << endl;

  cout << "total execution time: " << cl_comm.reduce_times_ms(execution_time) << " ms" << endl;

  cout << "average execution time: " << cl_comm.get_average_time_ms(execution_time) << " ms" << endl;

  cout << "average end to end: " << cl_comm.get_average_time_ms(execution_time) + cl_comm.get_average_time_ms(response_time)  << " ms" << endl;

  cout << "check value is: " << *(s_ctx.check_value) << endl;

  // if (strcmp("", FLAGS_output.c_str()) != 0) {
  //   cout << "outputing solution to " << FLAGS_output << endl;
  //   exec.exec_queue.enqueueReadBuffer(color_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, color);
  //   FILE * fp = fopen(FLAGS_output.c_str(), "w");
  //   if (!fp) { printf("ERROR: unable to open file %s\n", FLAGS_output.c_str()); }

  //   for (int i = 0; i < num_nodes; i++)
  //     fprintf(fp, "%d: %d\n", i + 1, color[i]);

  //   fclose(fp);

  // }

  free_scheduler_ctx(&exec, &s_ctx);
  // free(color);
  // free(node_value);

  return 0;
}
