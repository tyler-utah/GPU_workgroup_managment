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
DEFINE_int32(workgroups, 7, "Number of workgroups");
DEFINE_string(kernel_file, "test_suite/connect_four/device/connect_four.cl", "Kernel file name");

// FIXME: size number of pools w.r.t. occupancy
DEFINE_int32(pools, 10, "Number of task pools");
// Pool size is quite arbitrary, may be tuned
DEFINE_int32(pool_size, 20, "Size of a task pool");
DEFINE_int32(maxlevel, 2, "Max level of look-ahead");

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

  if (FLAGS_maxlevel > MAX_LOOKAHEAD) {
    cout << "Warning, lowering lookahead to " << MAX_LOOKAHEAD << endl;
    FLAGS_maxlevel = MAX_LOOKAHEAD;
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
  cl::Buffer d_board(exec.exec_context, CL_MEM_READ_WRITE, board_mem_size);
  err = exec.exec_queue.enqueueWriteBuffer(d_board, CL_TRUE, 0, board_mem_size, h_board);
  check_ocl(err);

  // nodes
  cl::Buffer d_nodes;
  d_nodes = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, NUM_NODE * sizeof(Node));
  err = exec.exec_queue.enqueueFillBuffer(d_nodes, 0, 0, NUM_NODE * sizeof(Node));
  check_ocl(err);

  cl::Buffer d_node_head;
  d_node_head = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
  err = exec.exec_queue.enqueueFillBuffer(d_node_head, 0, 0, sizeof(cl_int));
  check_ocl(err);

  // task pools
  const size_t task_pool_size = FLAGS_pools * FLAGS_pool_size * sizeof(Task);
  cl::Buffer d_task_pool;
  d_task_pool = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, task_pool_size);
  Task *h_task_pool = (Task *)calloc(FLAGS_pools * FLAGS_pool_size, sizeof(Task));
  if (h_task_pool == NULL) {
    cout << "calloc failed" << endl;
    exit(EXIT_FAILURE);
  }
  err = exec.exec_queue.enqueueWriteBuffer(d_task_pool, CL_TRUE, 0, task_pool_size, h_task_pool);
  check_ocl(err);

  cl::Buffer d_task_pool_lock;
  d_task_pool_lock = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, FLAGS_pools * sizeof(cl_int));
  err = exec.exec_queue.enqueueFillBuffer(d_task_pool_lock, 0, 0, FLAGS_pools * sizeof(cl_int));
  check_ocl(err);

  cl::Buffer d_task_pool_head;
  d_task_pool_head = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, FLAGS_pools * sizeof(cl_int));
  err = exec.exec_queue.enqueueFillBuffer(d_task_pool_head, 0, 0, FLAGS_pools * sizeof(cl_int));
  check_ocl(err);

  cl::Buffer d_next_move_value(exec.exec_context, CL_MEM_READ_WRITE, NUM_COL * sizeof(cl_int));
  err = exec.exec_queue.enqueueFillBuffer(d_next_move_value, 0, 0, NUM_COL * sizeof(cl_int));
  check_ocl(err);

  cl::Buffer d_root_done(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
  err = exec.exec_queue.enqueueFillBuffer(d_root_done, 0, 0, sizeof(cl_int));
  check_ocl(err);

  cl::Buffer d_debug_int;
  d_debug_int = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));

  cl::Buffer d_debug_board(exec.exec_context, CL_MEM_READ_WRITE, board_mem_size);
  err = exec.exec_queue.enqueueFillBuffer(d_debug_board, EMPTY, 0, board_mem_size);
  check_ocl(err);

  // Set args

  int arg_index = 0;
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_board));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, FLAGS_maxlevel));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_nodes));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_node_head));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_task_pool));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_task_pool_lock));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_task_pool_head));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, FLAGS_pools));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, FLAGS_pool_size));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_next_move_value));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_root_done));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_debug_int));
  check_ocl(exec.exec_kernels["connect_four"].setArg(arg_index++, d_debug_board));

  cout << "Args are set, run kernel (" << FLAGS_threads << " threads, " << FLAGS_workgroups << " workgroups)" << endl;

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

  Node h_nodes[NUM_NODE];
  err = exec.exec_queue.enqueueReadBuffer(d_nodes, CL_TRUE, 0, NUM_NODE * sizeof(Node), h_nodes);
  check_ocl(err);

  cout << "Nodes (limited to 70): " << endl;
  for (int i = 0; i < 70; i++) {
    printf("   [%2.2d]", i);
    printf(" l:%+d", h_nodes[i].level);
    printf(" p:%+2.2d", h_nodes[i].parent);
    printf(" v:%+3.3d", h_nodes[i].value);
    // moves
    printf(" m[");
    for (int j = 0; j < h_nodes[i].level; j++) {
      printf(" %d", h_nodes[i].moves[j]);
    }
    printf(" ]\n");
  }
  cout << endl;

  // print task pools and head
  err = exec.exec_queue.enqueueReadBuffer(d_task_pool, CL_TRUE, 0, FLAGS_pools * FLAGS_pool_size * sizeof(Task), h_task_pool);
  check_ocl(err);

  cl_int *h_task_pool_head = (cl_int *)calloc(FLAGS_pools, sizeof(cl_int));
  if (h_task_pool_head == NULL) {
    cout << "calloc failed" << endl;
    exit(EXIT_FAILURE);
  }
  err = exec.exec_queue.enqueueReadBuffer(d_task_pool_head, CL_TRUE, 0, FLAGS_pools * sizeof(cl_int), h_task_pool_head);
  check_ocl(err);

  int limit = FLAGS_pool_size < 20 ? FLAGS_pool_size : 20;
  printf("Pools (size limited to %d):\n", limit);
  for (int i = 0; i < FLAGS_pools; i++) {
    printf("Pool %2.2d (head %2.2d): ", i, h_task_pool_head[i]);
    for (int j = 0; j < limit; j++) {
      printf("%s", (j == h_task_pool_head[i]) ? "|" : " ");
      printf("%3.3d", h_task_pool[(i * FLAGS_pool_size) + j]);
    }
    if (FLAGS_pool_size >= limit) {
      printf(" ...");
    }
    printf("\n");
  }

  // Next move value
  cl_int *h_next_move_value = (cl_int *)calloc(NUM_COL, sizeof(cl_int));
  if (h_next_move_value == NULL) {
    cout << "calloc failed" << endl;
    exit(EXIT_FAILURE);
  }
  err = exec.exec_queue.enqueueReadBuffer(d_next_move_value, CL_TRUE, 0, NUM_COL * sizeof(cl_int), h_next_move_value);
  check_ocl(err);
  printf("Next move value:");
  for (int i = 0; i < NUM_COL; i++) {
    printf(" %d: %+3.3d", i, h_next_move_value[i]);
  }
  printf("\n");

  // Root done
  cl_int h_root_done = 0;
  err = exec.exec_queue.enqueueReadBuffer(d_root_done, CL_TRUE, 0, sizeof(cl_int), &h_root_done);
  check_ocl(err);
  printf("Root done: %d\n", h_root_done);

  // Timing
  cl_ulong kernel_start_ns, kernel_end_ns, kernel_time_ns;
  err = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &kernel_start_ns);
  check_ocl(err);
  err = event.getProfilingInfo(CL_PROFILING_COMMAND_END, &kernel_end_ns);
  check_ocl(err);
  kernel_time_ns = kernel_end_ns - kernel_start_ns;
  cout << "Kernel executed in " << kernel_time_ns << " ns" << endl;

  // debug
  err = exec.exec_queue.enqueueReadBuffer(d_debug_board, CL_TRUE, 0, board_mem_size, h_board);
  check_ocl(err);
  cout << "Debug board" << endl;
  print_board(h_board);

  cl_int h_debug_int;
  err = exec.exec_queue.enqueueReadBuffer(d_debug_int, CL_TRUE, 0, sizeof(cl_int), &h_debug_int);
  check_ocl(err);

  cout << "Debug int: " << h_debug_int << endl;

  // clean

  //----------------------------------------------------------------------

  cout << "End" << endl;
  exit(EXIT_SUCCESS);
}

/*---------------------------------------------------------------------------*/
