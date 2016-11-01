
DEFINE_string(non_persistent_kernel_file, "non_persistent_kernels/matmult/device/matmult.cl", "the path the non persistent file");

const int a_num_line = 100;
const int a_num_col = 100;
size_t a_mem_size;
cl::Buffer d_A;

const int b_num_line = 100;
const int b_num_col = 100;
size_t b_mem_size;
cl::Buffer d_B;

size_t c_mem_size;
cl::Buffer d_C;
cl_ulong *svm_c_hash;

/*---------------------------------------------------------------------------*/

const char* non_persistent_app_name() {
  return "matmult";
}

/*---------------------------------------------------------------------------*/

const char* non_persistent_kernel_name() {
  return "matmult";
}

/*---------------------------------------------------------------------------*/

void init_non_persistent_app(CL_Execution *exec) {

  cl_int err = 0;

  /* Matrix A */
  a_mem_size = sizeof(cl_int) * a_num_line * a_num_col;
  d_A = cl::Buffer(exec->exec_context, CL_MEM_READ_ONLY, a_mem_size);
  cl_int *h_A = (cl_int *)malloc(a_mem_size);
  if (h_A == NULL) {
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < (a_num_line * a_num_col); i++) {
    h_A[i] = i % 10;
  }
  err = exec->exec_queue.enqueueWriteBuffer(d_A, CL_TRUE, 0, a_mem_size, h_A);
  check_ocl(err);

  /* Matrix B */
  b_mem_size = sizeof(cl_int) * b_num_line * b_num_col;
  d_B = cl::Buffer(exec->exec_context, CL_MEM_READ_ONLY, b_mem_size);
  cl_int *h_B = (cl_int *)malloc(b_mem_size);
  if (h_B == NULL) {
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < (b_num_line * b_num_col); i++) {
    h_B[i] = i % 10;
  }
  err = exec->exec_queue.enqueueWriteBuffer(d_B, CL_TRUE, 0, b_mem_size, h_B);
  check_ocl(err);

  /* Matrix C */
  c_mem_size = sizeof(cl_int) * a_num_line * b_num_col;
  d_C = cl::Buffer(exec->exec_context, CL_MEM_WRITE_ONLY, c_mem_size);
  /* Hugues: not sure about the alignment value '4' here */
  svm_c_hash = (cl_ulong *) clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_ulong), 4);
  *svm_c_hash = 0;
}

/*---------------------------------------------------------------------------*/

int set_non_persistent_app_args(int arg_index, cl::Kernel k) {
  // Set the args for graphics kernel
  check_ocl(k.setArg(arg_index++, d_A));
  check_ocl(k.setArg(arg_index++, a_num_line));
  check_ocl(k.setArg(arg_index++, a_num_col));
  check_ocl(k.setArg(arg_index++, d_B));
  check_ocl(k.setArg(arg_index++, b_num_line));
  check_ocl(k.setArg(arg_index++, b_num_col));
  check_ocl(k.setArg(arg_index++, d_C));
  check_ocl(clSetKernelArgSVMPointer(k(), arg_index++, svm_c_hash));

  return arg_index;
}

/*---------------------------------------------------------------------------*/

void reset_non_persistent() {
  *svm_c_hash = 0;
  return;
}

/*---------------------------------------------------------------------------*/

bool check_non_persistent_task() {
  printf ("matmult: svm_c_hash: %lu\n", *svm_c_hash);
  return *svm_c_hash == 3310879381;
}

/*---------------------------------------------------------------------------*/

void clean_non_persistent_task(CL_Execution *exec) {
  clSVMFree(exec->exec_context(), svm_c_hash);
}

/*---------------------------------------------------------------------------*/
