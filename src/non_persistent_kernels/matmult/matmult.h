
DEFINE_string(non_persistent_kernel_file, "non_persistent_kernels/matmult/device/matmult.cl", "the path the non persistent file");

DEFINE_int32(seed, 1234, "Seed for pseudo-random number generator to fill matrices");
DEFINE_int32(A_row, 100, "Number of row for matrix A");
DEFINE_int32(A_col, 100, "Number of col for matrix A");
DEFINE_int32(B_row, 100, "Number of row for matrix B");
DEFINE_int32(B_col, 100, "Number of col for matrix B");
DEFINE_int32(matdim, 0, "Number of row and col for both input matrixes");

size_t a_mem_size;
cl::Buffer d_A;

size_t b_mem_size;
cl::Buffer d_B;

size_t c_mem_size;
cl_int *svm_C;

int ref_hash;
cl_int *svm_counter;
cl_int *svm_hash;

/*---------------------------------------------------------------------------*/

const char* non_persistent_app_name() {
  return "matmult";
}

/*---------------------------------------------------------------------------*/

const char* non_persistent_kernel_name() {
  return "matmult";
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

void host_matrix_multiply(cl_int *A, int A_row, int A_col,
                         cl_int *B, int B_row, int B_col,
                         cl_int *C)
{
  int C_row = A_row;
  int C_col = B_col;

  if (A_col != B_row) {
    return;
  }

  for (int r = 0 ; r < C_row; r++) {
    for (int c = 0; c < C_col; c++) {
      C[(r * C_row) + c] = 0;
      for (int i = 0; i < A_col; i++) {
        C[(r * C_row) + c] += A[(r * A_row) + i] * B[(i * B_row) + c];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

void init_non_persistent_app(CL_Execution *exec) {
  cl_int err = 0;
  int seed = FLAGS_seed;

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

  /* Matrix A */
  a_mem_size = sizeof(cl_int) * FLAGS_A_row * FLAGS_A_col;
  d_A = cl::Buffer(exec->exec_context, CL_MEM_READ_ONLY, a_mem_size);
  cl_int *h_A = (cl_int *)malloc(a_mem_size);
  if (h_A == NULL) {
    exit(EXIT_FAILURE);
  }
  seed = rand_fill(h_A, FLAGS_A_row, FLAGS_A_col, seed);
  err = exec->exec_queue.enqueueWriteBuffer(d_A, CL_TRUE, 0, a_mem_size, h_A);
  check_ocl(err);

  /* Matrix B */
  b_mem_size = sizeof(cl_int) * FLAGS_B_row * FLAGS_B_col;
  d_B = cl::Buffer(exec->exec_context, CL_MEM_READ_ONLY, b_mem_size);
  cl_int *h_B = (cl_int *)malloc(b_mem_size);
  if (h_B == NULL) {
    exit(EXIT_FAILURE);
  }
  seed = rand_fill(h_B, FLAGS_B_row, FLAGS_B_col, seed);
  err = exec->exec_queue.enqueueWriteBuffer(d_B, CL_TRUE, 0, b_mem_size, h_B);
  check_ocl(err);

  /* Matrix C */
  c_mem_size = sizeof(cl_int) * FLAGS_A_row * FLAGS_B_col;
  svm_C = (cl_int *)clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, c_mem_size, 4);
  if (svm_C == NULL) {
    printf("clSVMAlloc failed\n");
    exit(EXIT_FAILURE);
  }

  /* Compute C to obtain hash */
  host_matrix_multiply(h_A, FLAGS_A_row, FLAGS_A_col,
                       h_B, FLAGS_B_row, FLAGS_B_col,
                       svm_C);

  ref_hash = hash_mat(svm_C, FLAGS_A_row, FLAGS_B_col);

  memset(svm_C, 0, c_mem_size);

  /* counter and hash */
  svm_counter = (cl_int *)clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
  svm_hash = (cl_int *)clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);

  *svm_counter = 0;
  *svm_hash = 0;

  free(h_A);
  free(h_B);
}

/*---------------------------------------------------------------------------*/

int set_non_persistent_app_args(int arg_index, cl::Kernel k) {
  // Set the args for graphics kernel
  check_ocl(k.setArg(arg_index++, d_A));
  check_ocl(k.setArg(arg_index++, FLAGS_A_row));
  check_ocl(k.setArg(arg_index++, FLAGS_A_col));
  check_ocl(k.setArg(arg_index++, d_B));
  check_ocl(k.setArg(arg_index++, FLAGS_B_row));
  check_ocl(k.setArg(arg_index++, FLAGS_B_col));
  check_ocl(clSetKernelArgSVMPointer(k(), arg_index++, svm_C));
  check_ocl(clSetKernelArgSVMPointer(k(), arg_index++, svm_counter));
  check_ocl(clSetKernelArgSVMPointer(k(), arg_index++, svm_hash));
  return arg_index;
}

/*---------------------------------------------------------------------------*/

void reset_non_persistent() {
  *svm_counter = 0;
  *svm_hash = 0;
  memset(svm_C, 0, c_mem_size);
}

/*---------------------------------------------------------------------------*/

bool check_non_persistent_task() {
  return *svm_hash == ref_hash;
}

/*---------------------------------------------------------------------------*/

void clean_non_persistent_task(CL_Execution *exec) {
  clSVMFree(exec->exec_context(), svm_C);
  clSVMFree(exec->exec_context(), svm_hash);
  clSVMFree(exec->exec_context(), svm_counter);
}

/*---------------------------------------------------------------------------*/
