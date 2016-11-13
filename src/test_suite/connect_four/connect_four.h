#include "connect_four_data.h"
#include <math.h>

/*---------------------------------------------------------------------------*/

DEFINE_string(restoration_ctx_path, "test_suite/connect_four", "Path to restoration context");
DEFINE_string(merged_kernel_file, "test_suite/connect_four/device/merged.cl", "path to the merged mega kernel file");
DEFINE_string(persistent_kernel_file, "test_suite/connect_four/device/standalone.cl", "path to the standalone mega kernel file");

/*---------------------------------------------------------------------------*/

// Pool size is quite arbitrary, may be tuned
DEFINE_int32(pool_size, 10000, "Size of a task pool");
DEFINE_int32(maxlevel, 2, "Max level of look-ahead");

DEFINE_string(board_file, "test_suite/connect_four/board.txt", "Path to board description");

/*---------------------------------------------------------------------------*/
// global vars

cl_uchar *h_board;
cl::Buffer d_board;
Node *h_nodes;
cl::Buffer d_nodes;
cl::Buffer d_node_head;
cl_int num_node;
cl::Buffer d_task_pool;
Task *h_task_pool;
cl::Buffer d_task_pool_lock;
cl::Buffer d_task_pool_head;
cl::Buffer d_next_move_value;
cl::Buffer d_root_done;
cl::Buffer d_debug_int;
cl::Buffer d_debug_board;
size_t board_mem_size;
cl_int num_workgroups;

/*===========================================================================*/
// helper functions

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

int compute_num_node(int maxlevel)
{
  /* overflow safety: add an extra node */
  int sum = 1;
  for (int i = 1; i <= maxlevel; i++) {
    sum += pow(7, i);
  }
  return sum;
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

/*===========================================================================*/

const char* persistent_app_name() {
  return "connect_four";
}

/*---------------------------------------------------------------------------*/

const char* persistent_kernel_name() {
  return "connect_four";
}
/*---------------------------------------------------------------------------*/

void init_persistent_app_for_occupancy(CL_Execution *exec)
{
  // nothing to do
}

/*---------------------------------------------------------------------------*/

int set_persistent_app_args_for_occupancy(int arg_index, cl::Kernel k) {
  // Set dummy args for persistent kernel

  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, 0));
  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, 0));
  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, 0));
  check_ocl(k.setArg(arg_index++, 0));
  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, NULL));

  return arg_index;
}

/*---------------------------------------------------------------------------*/

void reset_persistent_task(CL_Execution *exec) {
  // re-write 0 to the CL buffers, etc
  cl_int err = 0;

  // board
  FILE *f = fopen(file::Path(FLAGS_board_file), "r");
  if (f == NULL) {
    cout << "Could not open board file: " << FLAGS_board_file << endl;
    exit(EXIT_FAILURE);
  }
  scan_board(f, h_board);
  fclose(f);

  err = exec->exec_queue.enqueueWriteBuffer(d_board, CL_TRUE, 0, board_mem_size, h_board);
  check_ocl(err);

  // nodes
  err = exec->exec_queue.enqueueFillBuffer(d_nodes, 0, 0, num_node * sizeof(Node));
  check_ocl(err);
  err = exec->exec_queue.enqueueFillBuffer(d_node_head, 0, 0, sizeof(cl_int));
  check_ocl(err);

  // tasks
  h_task_pool = (Task *)calloc(num_workgroups * FLAGS_pool_size, sizeof(Task));
  if (h_task_pool == NULL) {
    cout << "calloc failed" << endl;
    exit(EXIT_FAILURE);
  }
  err = exec->exec_queue.enqueueWriteBuffer(d_task_pool, CL_TRUE, 0, num_workgroups * FLAGS_pool_size * sizeof(Task), h_task_pool);
  check_ocl(err);

  err = exec->exec_queue.enqueueFillBuffer(d_task_pool_lock, 0, 0, num_workgroups * sizeof(cl_int));
  check_ocl(err);

  err = exec->exec_queue.enqueueFillBuffer(d_task_pool_head, 0, 0, num_workgroups * sizeof(cl_int));
  check_ocl(err);

  // others
  err = exec->exec_queue.enqueueFillBuffer(d_next_move_value, 0, 0, NUM_COL * sizeof(cl_int));
  check_ocl(err);

  err = exec->exec_queue.enqueueFillBuffer(d_root_done, 0, 0, sizeof(cl_int));
  check_ocl(err);

  err = exec->exec_queue.enqueueFillBuffer(d_debug_board, EMPTY, 0, board_mem_size);
  check_ocl(err);

}

/*---------------------------------------------------------------------------*/

void init_persistent_app_for_real(CL_Execution *exec, int occupancy) {
  cl_int err = 0;
  num_workgroups = occupancy;

  // board
  board_mem_size = NUM_CELL * sizeof(cl_uchar);
  h_board = (cl_uchar *)calloc(NUM_CELL, sizeof(cl_uchar));
  if (h_board == NULL) {
    cout << "calloc failed" << endl;
    exit(EXIT_FAILURE);
  }
  d_board = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, board_mem_size);

  // nodes
  num_node = compute_num_node(FLAGS_maxlevel);
  cout << "Num node: " << num_node << endl;
  d_nodes = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_node * sizeof(Node));
  d_node_head = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));

  // tasks
  d_task_pool = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_workgroups * FLAGS_pool_size * sizeof(Task));
  d_task_pool_lock = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_workgroups * sizeof(cl_int));
  d_task_pool_head = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_workgroups * sizeof(cl_int));

  // others
  d_next_move_value = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, NUM_COL * sizeof(cl_int));
  d_root_done = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
  d_debug_int = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
  d_debug_board = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, board_mem_size);

  reset_persistent_task(exec);
}

/*---------------------------------------------------------------------------*/

void set_persistent_app_args_for_real(int arg_index, cl::Kernel k) {
  // Set args for persistent kernel
  check_ocl(k.setArg(arg_index++, d_board));
  check_ocl(k.setArg(arg_index++, FLAGS_maxlevel));
  check_ocl(k.setArg(arg_index++, d_nodes));
  check_ocl(k.setArg(arg_index++, d_node_head));
  check_ocl(k.setArg(arg_index++, num_node));
  check_ocl(k.setArg(arg_index++, d_task_pool));
  check_ocl(k.setArg(arg_index++, d_task_pool_lock));
  check_ocl(k.setArg(arg_index++, d_task_pool_head));
  check_ocl(k.setArg(arg_index++, num_workgroups));
  check_ocl(k.setArg(arg_index++, FLAGS_pool_size));
  check_ocl(k.setArg(arg_index++, d_next_move_value));
  check_ocl(k.setArg(arg_index++, d_root_done));
  check_ocl(k.setArg(arg_index++, d_debug_int));
  check_ocl(k.setArg(arg_index++, d_debug_board));
}

/*---------------------------------------------------------------------------*/

void output_persistent_solution(const char *fname, CL_Execution *exec) {
  // write to a file, nothing to do for connect four
  return;
}

/*---------------------------------------------------------------------------*/

void clean_persistent_task(CL_Execution *exec) {
  free(h_task_pool);
}

/*---------------------------------------------------------------------------*/

bool diff_solution_file_int(int * a, const char * solution_fname, int v) {
  // pannotia specific
  return 0;
}

/*---------------------------------------------------------------------------*/

bool check_persistent_task(CL_Execution *exec) {
  // check whether the output is correct, load cl buffer back in host
  // and check values
  cl_int err = 0;
  int limit = 0;

  // Original board
  cout << "Base board" << endl;
  print_board(h_board);

  // Nodes
  // Node *h_nodes = (Node *)calloc(num_node, sizeof(Node));
  // if (h_nodes == NULL) {
  //   cout << "calloc failed" << endl;
  //   exit(EXIT_FAILURE);
  // }
  // err = exec->exec_queue.enqueueReadBuffer(d_nodes, CL_TRUE, 0, num_node * sizeof(Node), h_nodes);
  // check_ocl(err);

  // limit = num_node < 70 ? num_node : 70;
  // cout << "Nodes (limited to " << limit << "): " << endl;
  // for (int i = 0; i < limit; i++) {
  //   printf("   [%2.2d]", i);
  //   printf(" l:%+d", h_nodes[i].level);
  //   printf(" p:%+2.2d", h_nodes[i].parent);
  //   printf(" v:%+3.3d", h_nodes[i].value);
  //   // moves
  //   printf(" m[");
  //   for (int j = 0; j < h_nodes[i].level; j++) {
  //     printf(" %d", h_nodes[i].moves[j]);
  //   }
  //   printf(" ]\n");
  // }
  // cout << endl;

  // print task pools and head
  // err = exec->exec_queue.enqueueReadBuffer(d_task_pool, CL_TRUE, 0, num_workgroups * FLAGS_pool_size * sizeof(Task), h_task_pool);
  // check_ocl(err);

  // cl_int *h_task_pool_head = (cl_int *)calloc(num_workgroups, sizeof(cl_int));
  // if (h_task_pool_head == NULL) {
  //   cout << "calloc failed" << endl;
  //   exit(EXIT_FAILURE);
  // }
  // err = exec->exec_queue.enqueueReadBuffer(d_task_pool_head, CL_TRUE, 0, num_workgroups * sizeof(cl_int), h_task_pool_head);
  // check_ocl(err);

  // limit = FLAGS_pool_size < 20 ? FLAGS_pool_size : 20;
  // printf("Pools (size limited to %d):\n", limit);
  // for (int i = 0; i < num_workgroups; i++) {
  //   printf("Pool %2.2d (head %2.2d): ", i, h_task_pool_head[i]);
  //   for (int j = 0; j < limit; j++) {
  //     printf("%s", (j == h_task_pool_head[i]) ? "|" : " ");
  //     printf("%3.3d", h_task_pool[(i * FLAGS_pool_size) + j]);
  //   }
  //   if (FLAGS_pool_size >= limit) {
  //     printf(" ...");
  //   }
  //   printf("\n");
  // }

  // Next move value
  cl_int *h_next_move_value = (cl_int *)calloc(NUM_COL, sizeof(cl_int));
  if (h_next_move_value == NULL) {
    cout << "calloc failed" << endl;
    exit(EXIT_FAILURE);
  }
  err = exec->exec_queue.enqueueReadBuffer(d_next_move_value, CL_TRUE, 0, NUM_COL * sizeof(cl_int), h_next_move_value);
  check_ocl(err);
  printf("Next move value:");
  for (int i = 0; i < NUM_COL; i++) {
    printf(" %d: %+3.3d", i, h_next_move_value[i]);
  }
  printf("\n");

  // check values for the board committed in the git repo
  bool check = false;
  check |= h_next_move_value[0] == (cl_int)0;
  check |= h_next_move_value[1] == (cl_int)1;
  check |= h_next_move_value[2] == (cl_int)PLUS_INF;
  check |= h_next_move_value[3] == (cl_int)2;
  check |= h_next_move_value[4] == (cl_int)2;
  check |= h_next_move_value[5] == (cl_int)0;
  check |= h_next_move_value[6] == (cl_int)0;

  // Root done
  cl_int h_root_done = 0;
  err = exec->exec_queue.enqueueReadBuffer(d_root_done, CL_TRUE, 0, sizeof(cl_int), &h_root_done);
  check_ocl(err);
  printf("Root done: %d\n", h_root_done);

  // debug
  // err = exec->exec_queue.enqueueReadBuffer(d_debug_board, CL_TRUE, 0, board_mem_size, h_board);
  // check_ocl(err);
  // cout << "Debug board" << endl;
  // print_board(h_board);

  // cl_int h_debug_int;
  // err = exec->exec_queue.enqueueReadBuffer(d_debug_int, CL_TRUE, 0, sizeof(cl_int), &h_debug_int);
  // check_ocl(err);

  // cout << "Debug int: " << h_debug_int << endl;

  return check;
}

/*---------------------------------------------------------------------------*/
