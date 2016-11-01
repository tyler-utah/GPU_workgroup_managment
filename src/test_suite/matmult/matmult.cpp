#include <iostream>
#include <vector>

#include "base/commandlineflags.h"
#include "base/file.h"
#include "opencl/opencl.h"
#include "cl_execution.h"

/*---------------------------------------------------------------------------*/

DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_int32(threads, 128, "Number of threads per workgroups");
DEFINE_int32(workgroups, 4, "Number of workgroups");
DEFINE_string(kernel_file, "test_suite/matmult/matmult.cl", "Kernel file name");

// dummy needed to fill cl_execution primitives args
DEFINE_string(scheduler_rt_path, "scheduler_rt/rt_device", "Dummy");
DEFINE_string(restoration_ctx_path, "test_suite/os_freq/common/", "Dummy (Path to restoration context)");

/*---------------------------------------------------------------------------*/

using namespace std;

/*---------------------------------------------------------------------------*/

void print_mat(char *name, cl_int *M, int m_num_line, int m_num_col, int limit)
{
  int l, c;
  cout << "Matrix " << name << ":" << endl;
  for (l = 0; l < m_num_line; l++) {
    for (c = 0; c < m_num_col; c++) {
      printf(" %2.2d", M[(l * m_num_line) + c]);
      if (c >= limit) {
        printf (" ...");
        break;
      }
    }
    printf("\n");
    if (l >= limit) {
      printf (" ...\n");
      break;
    }
  }
}

/*---------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
  cout << "Start" << endl;

  flags::ParseCommandLineFlags(&argc, &argv, true);

  cl_int err;

  // Platforms
  vector<cl::Platform> platforms;
  err = cl::Platform::get(&platforms);
  check_ocl(err);

  // Devices
  vector<vector<cl::Device>> devices;
  for (unsigned int i = 0; i < platforms.size(); i++) {
    vector<cl::Device> tmp_devices;
    err = platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &tmp_devices);
    check_ocl(err);
    devices.push_back(tmp_devices);
  }

  // Execution environment
  CL_Execution exec;
  exec.exec_device = devices[FLAGS_platform_id][FLAGS_device_id];
  cout << "Using GPU: " << exec.getExecDeviceName().c_str() << endl;

  // Context and queue
  cl::Context context(exec.exec_device);
  exec.exec_context = context;
  cl::CommandQueue queue(exec.exec_context, CL_QUEUE_PROFILING_ENABLE);
  exec.exec_queue = queue;

  // compile kernel
  cout << "Compile kernel file: " << FLAGS_kernel_file << endl;
  err = exec.compile_kernel(file::Path(FLAGS_kernel_file),
                            file::Path(FLAGS_scheduler_rt_path),
                            file::Path(FLAGS_restoration_ctx_path),
                            0);
  check_ocl(err);
  exec.exec_kernels["matmult"] = cl::Kernel(exec.exec_program, "matmult", &err);
  check_ocl(err);

  //----------------------------------------------------------------------
  // specific to matmult

  // Kernel arguments
  const int a_num_line = 16;
  const int a_num_col = 16;
  const size_t a_mem_size = sizeof(cl_int) * a_num_line * a_num_col;
  cl::Buffer d_A(exec.exec_context, CL_MEM_READ_ONLY, a_mem_size);
  cl_int h_A[16*16] = {
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
  };
  err = exec.exec_queue.enqueueWriteBuffer(d_A, CL_TRUE, 0, a_mem_size, h_A);
  check_ocl(err);

  const int b_num_line = 16;
  const int b_num_col = 16;
  const size_t b_mem_size = sizeof(cl_int) * b_num_line * b_num_col;
  cl::Buffer d_B(exec.exec_context, CL_MEM_READ_ONLY, b_mem_size);
  cl_int h_B[16*16] = {
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
  };
  err = exec.exec_queue.enqueueWriteBuffer(d_B, CL_TRUE, 0, b_mem_size, h_B);
  check_ocl(err);

  const size_t c_mem_size = sizeof(cl_int) * a_num_line * b_num_col;
  cl::Buffer d_C(exec.exec_context, CL_MEM_WRITE_ONLY, c_mem_size);
  err = exec.exec_queue.enqueueFillBuffer(d_C, 0, 0, sizeof(cl_int));
  check_ocl(err);
  cl::Buffer d_c_hash(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_ulong));

  int arg_index = 0;
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, d_A));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, a_num_line));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, a_num_col));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, d_B));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, b_num_line));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, b_num_col));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, d_C));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, d_c_hash));

  // Run kernel
  cl::Event event;
  cl::NDRange global_size(FLAGS_workgroups * FLAGS_threads);
  cl::NDRange local_size(FLAGS_threads);
  exec.exec_queue.enqueueNDRangeKernel(exec.exec_kernels["matmult"],
                                       cl::NullRange,
                                       global_size,
                                       local_size,
                                       NULL,
                                       &event);

  event.wait();

  // Computation results

  // Timing
  cl_ulong kernel_start_ns, kernel_end_ns, kernel_time_ns;
  err = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &kernel_start_ns);
  check_ocl(err);
  err = event.getProfilingInfo(CL_PROFILING_COMMAND_END, &kernel_end_ns);
  check_ocl(err);
  kernel_time_ns = kernel_end_ns - kernel_start_ns;
  cout << "Kernel executed in " << kernel_time_ns << " ns" << endl;

  // Resulting matrix
  cl_int *h_C = (cl_int *) calloc(a_num_line * b_num_col, sizeof(cl_int));
  if (h_C == NULL) {
    cout << "calloc failed" << endl;
    exit(EXIT_FAILURE);
  }
  err = exec.exec_queue.enqueueReadBuffer(d_C, CL_TRUE, 0, c_mem_size, h_C);
  check_ocl(err);
  print_mat("C", h_C, a_num_line, b_num_col, 12);

  // Hash for matrix C
  cl_ulong h_c_hash = 0;
  err = exec.exec_queue.enqueueReadBuffer(d_c_hash, CL_TRUE, 0, sizeof(cl_ulong), &h_c_hash);
  check_ocl(err);

  cout << "Hash: " << h_c_hash << endl;

  // clean
  free(h_C);

  //----------------------------------------------------------------------

  cout << "End" << endl;
  exit(EXIT_SUCCESS);
}

/*---------------------------------------------------------------------------*/
