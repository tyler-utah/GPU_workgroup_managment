#include <iostream>
#include <vector>

#include "base/commandlineflags.h"
#include "base/file.h"
#include "opencl/opencl.h"
#include "cl_execution.h"

/*---------------------------------------------------------------------------*/

DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_int32(threads, 256, "Number of threads per workgroups");
DEFINE_int32(workgroups, 4, "Number of workgroups");
DEFINE_string(kernel_file, "non_persistent_kernels/matmult/device/matmult.cl", "Kernel file name");

DEFINE_int32(seed, 1234, "Seed for pseudo-random number generator to fill matrices");
DEFINE_int32(A_row, 100, "Number of row for matrix A");
DEFINE_int32(A_col, 100, "Number of col for matrix A");
DEFINE_int32(B_row, 100, "Number of row for matrix B");
DEFINE_int32(B_col, 100, "Number of col for matrix B");
DEFINE_int32(matdim, 0, "Number of row and col for both input matrixes");

// dummy needed to fill cl_execution primitives args
DEFINE_string(scheduler_rt_path, "scheduler_rt/rt_device", "Dummy");
DEFINE_string(restoration_ctx_path, "test_suite/os_freq/common/", "Dummy (Path to restoration context)");

/*---------------------------------------------------------------------------*/

using namespace std;

/*---------------------------------------------------------------------------*/

void print_mat(char *name, cl_int *M, int num_row, int num_col, int limit)
{
  int r, c;
  cout << "Matrix " << name;
  cout << " ( " << num_row << " x " << num_col;
  cout << " , print limited to ";
  cout << limit << " x " << limit << " ):" << endl;

  for (r = 0; r < num_row; r++) {
    if (r >= limit) {
      printf (" ...\n");
      break;
    }
    for (c = 0; c < num_col; c++) {
      if (c >= limit) {
        printf (" ...");
        break;
      }
      printf(" %+10.10d", M[(r * num_row) + c]);
    }
    printf("\n");
  }
}

/*---------------------------------------------------------------------------*/

int rand_fill(cl_int *M, int num_row, int num_col, int seed)
{
  // cheap random: middle square method
  for (int row = 0; row < num_row; row++) {
    for (int col = 0; col < num_col; col++) {
      seed = seed * seed;
      seed = (seed / 1000) % 1000000;
      M[(row * num_col) + col] = (cl_int)seed;
    }
  }

  return seed;
}

/*---------------------------------------------------------------------------*/

int hash_mat(cl_int *M, int num_row, int num_col)
{
  // hash the diagonal using djb2, see
  // http://www.cse.yorku.ca/~oz/hash.html
  int hash = 5381;
  int row = 0;
  int col = 0;
  while (row < num_row && col < num_col) {
    hash = (hash * 33) + M[(row * num_col) + col];
    row++;
    col++;
  }
  return hash;
}

/*---------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
  cout << "Start" << endl;

  flags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_matdim > 0) {
    FLAGS_A_row = FLAGS_matdim;
    FLAGS_A_col = FLAGS_matdim;
    FLAGS_B_row = FLAGS_matdim;
    FLAGS_B_col = FLAGS_matdim;
  }

  if (FLAGS_A_col != FLAGS_B_row) {
    cout << "Error: incompatile matrix size (A col: " << FLAGS_A_col;
    cout << ", B row: " << FLAGS_B_row << ")" << endl;
    exit(EXIT_FAILURE);
  }

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
  int seed = FLAGS_seed;
  cout << "Seed: " << seed << endl;

  cl_int *h_A = (cl_int *)calloc(FLAGS_A_row * FLAGS_A_col, sizeof(cl_int));
  if (h_A == NULL) {
    cout << "calloc failed" << endl;
  }
  seed = rand_fill(h_A, FLAGS_A_row, FLAGS_A_col, seed);
  print_mat("A", h_A, FLAGS_A_row, FLAGS_A_col, 9);

  const size_t a_mem_size = sizeof(cl_int) * FLAGS_A_row * FLAGS_A_col;
  cl::Buffer d_A(exec.exec_context, CL_MEM_READ_WRITE, a_mem_size);
  err = exec.exec_queue.enqueueWriteBuffer(d_A, CL_TRUE, 0, a_mem_size, h_A);
  check_ocl(err);

  cl_int *h_B = (cl_int *)calloc(FLAGS_B_row * FLAGS_B_col, sizeof(cl_int));
  if (h_B == NULL) {
    cout << "calloc failed" << endl;
  }
  seed = rand_fill(h_B, FLAGS_B_row, FLAGS_B_col, seed);
  print_mat("B", h_B, FLAGS_B_row, FLAGS_B_col, 9);

  const size_t b_mem_size = sizeof(cl_int) * FLAGS_B_row * FLAGS_B_col;
  cl::Buffer d_B(exec.exec_context, CL_MEM_READ_WRITE, b_mem_size);
  err = exec.exec_queue.enqueueWriteBuffer(d_B, CL_TRUE, 0, b_mem_size, h_B);
  check_ocl(err);

  const size_t c_mem_size = sizeof(cl_int) * FLAGS_A_row * FLAGS_B_col;
  cl::Buffer d_C(exec.exec_context, CL_MEM_READ_WRITE, c_mem_size);
  err = exec.exec_queue.enqueueFillBuffer(d_C, 0, 0, sizeof(cl_int));
  check_ocl(err);

  int arg_index = 0;
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, d_A));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, FLAGS_A_row));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, FLAGS_A_col));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, d_B));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, FLAGS_B_row));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, FLAGS_B_col));
  check_ocl(exec.exec_kernels["matmult"].setArg(arg_index++, d_C));

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
  cout << "Kernel executed in ";
  cout << (float)((float)kernel_time_ns / (float)1000000);
  cout << " ms" << endl;

  // Resulting matrix
  cl_int *h_C = (cl_int *) calloc(FLAGS_A_row * FLAGS_B_col, sizeof(cl_int));
  if (h_C == NULL) {
    cout << "calloc failed" << endl;
    exit(EXIT_FAILURE);
  }
  err = exec.exec_queue.enqueueReadBuffer(d_C, CL_TRUE, 0, c_mem_size, h_C);
  check_ocl(err);
  print_mat("C", h_C, FLAGS_A_row, FLAGS_B_col, 9);

  // Hash for matrix C
  int c_hash = hash_mat(h_C, FLAGS_A_row, FLAGS_B_col);

  cout << "Hash: " << c_hash << endl;

  // clean
  free(h_A);
  free(h_B);
  free(h_C);

  //----------------------------------------------------------------------

  cout << "End" << endl;
  exit(EXIT_SUCCESS);
}

/*---------------------------------------------------------------------------*/
