#include <iostream>
#include <vector>

#include "cl_types.h"
#include "base/commandlineflags.h"
#include "base/file.h"
#include "opencl/opencl.h"
#include "cl_execution.h"

#include "connect_four_data.h"

/*---------------------------------------------------------------------------*/

DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_int32(threads, 256, "Number of threads per workgroups");
DEFINE_int32(workgroups, 4, "Number of workgroups");
DEFINE_string(kernel_file, "test_suite/connect_four/device/connect_four.cl", "Kernel file name");

// dummy needed to fill cl_execution primitives args
DEFINE_string(scheduler_rt_path, "scheduler_rt/rt_device", "Dummy");
DEFINE_string(restoration_ctx_path, "test_suite/connect_four/common/", "Path to restoration context and other header files");

DEFINE_string(board_file, "test_suite/connect_four/board.txt", "Path to board description");

/*---------------------------------------------------------------------------*/

using namespace std;

/*---------------------------------------------------------------------------*/

void print_board(cl_uchar *board)
{
  printf(" - - - - - - -\n");

  for (int i = 0; i < NUM_ROW; i++) {
    for (int j = 0; j < NUM_COL; j++) {
      switch (board[(i * 7) + j]) {
      case EMPTY:
        printf(" .");
        break;
      case HUMAN:
        printf(" o");
        break;
      case COMPUTER:
        printf(" x");
        break;
      default:
        printf(" ?");
        break;
      }
    }
    printf("\n");
  }

  printf(" - - - - - - -\n");
}

/*---------------------------------------------------------------------------*/

void scan_board(FILE *f, cl_uchar *board)
{
  int c;
  int i = 0;
  do {
    c = fgetc(f);
    if (c == '.') {
      board[i++] = EMPTY;
    }
    if (c == 'o') {
      board[i++] = HUMAN;
    }
    if (c == 'x') {
      board[i++] = COMPUTER;
    }
  } while (c != EOF && i < NUM_CELL);

  if (i < NUM_CELL) {
    cout << "Not enough valid character in board" << endl;
    exit(EXIT_FAILURE);
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
  exec.exec_kernels["connect_four"] = cl::Kernel(exec.exec_program, "connect_four", &err);
  check_ocl(err);

  //----------------------------------------------------------------------
  // specific to connect_four

  // Board is 6 lines * 7 columns, see dimensions in connect_four_data.h
  cl_uchar h_board[NUM_CELL] = {EMPTY};
  FILE *f = fopen(file::Path(FLAGS_board_file), "r");
  if (f == NULL) {
    cout << "Could not open board file: " << FLAGS_board_file << endl;
    exit(EXIT_FAILURE);
  }
  scan_board(f, h_board);
  fclose(f);
  cout << "Base board" << endl;
  print_board(h_board);

  const size_t board_mem_size = sizeof(cl_uchar) * 6 * 7;
  cl::Buffer d_board(exec.exec_context, CL_MEM_READ_ONLY, board_mem_size);
  err = exec.exec_queue.enqueueWriteBuffer(d_board, CL_TRUE, 0, board_mem_size, h_board);
  check_ocl(err);

  cl_int h_value = 0;
  cl::Buffer d_value(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
  err = exec.exec_queue.enqueueWriteBuffer(d_value, CL_TRUE, 0, sizeof(cl_int), &h_value);
  check_ocl(err);

  // Set args

  int arg_index = 0;
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_board));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_value));

  // Run kernel
  cl::Event event;
  cl::NDRange global_size(FLAGS_workgroups * FLAGS_threads);
  cl::NDRange local_size(FLAGS_threads);
  exec.exec_queue.enqueueNDRangeKernel(exec.exec_kernels["connect_four"],
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

  // Resulting value
  err = exec.exec_queue.enqueueReadBuffer(d_board, CL_TRUE, 0, board_mem_size, h_board);
  check_ocl(err);
  cout << "Updated board" << endl;
  print_board(h_board);

  err = exec.exec_queue.enqueueReadBuffer(d_value, CL_TRUE, 0, sizeof(cl_int), &h_value);
  check_ocl(err);

  cout << "Board value: ";
  if (h_value == PLUS_INF) {
    cout << "PLUS_INF (computer wins)" ;
  } else if (h_value == MINUS_INF) {
    cout << "MINUS_INF (human wins)" ;
  } else {
    cout << h_value;
  }
  cout << endl;

  // clean

  //----------------------------------------------------------------------

  cout << "End" << endl;
  exit(EXIT_SUCCESS);
}

/*---------------------------------------------------------------------------*/
