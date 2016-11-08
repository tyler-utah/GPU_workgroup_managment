#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

/*
  Matrix multiplication: C = A * B

  We assume C is big enough to store the result.
*/

/*---------------------------------------------------------------------------*/

int hash_mat(int *M, int num_row, int num_col) {
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

void matmult(__global int *A, const int A_row, const int A_col, __global int *B,
             const int B_row, const int B_col, __global int *C,
             __global atomic_int *counter, __global atomic_int *hash,
             __global Kernel_ctx *__k_ctx) {
  /* safety */
  if (A_col != B_row) {
    return;
  }

  int gid = k_get_global_id(__k_ctx);
  int num_threads = k_get_global_size(__k_ctx);
  int c_size = A_row * B_col;

  /* Multiply matrices */
  for (int i = gid; i < c_size; i += num_threads) {
    int c_row = i / B_col;
    int c_col = i % B_col;
    int a_offset = c_row * A_col;
    C[i] = 0;
    for (int j = 0; j < B_row; j++) {
      C[i] += A[a_offset + j] * B[(j * B_col) + c_col];
    }
  }

  if (get_local_id(0) == 0) {
    int finished = atomic_fetch_add(counter, 1);
    if (finished == (k_get_num_groups(__k_ctx) - 1)) {
      int h = hash_mat(C, A_row, B_col);
      atomic_store(hash, h);
    }
  }
}

/*---------------------------------------------------------------------------*/

__global int __junk_global;

// This kernel is the naive version of the mega-kernel using
// the global barrier

// Tyler: Notes: Simply adding the scheduler args in the
// standalone application causes a massive slowdown
// (~13 seconds to ~24 seconds).
// But the merged kernel *necessarily* must have
// the scheduler context, so it isn't really fair
// to compare without it.
void sssp_combined(const int num_rows, __global int *row, __global int *col,
                   __global int *data, __global int *x, __global int *y,
                   __global int *stop, __global IW_barrier *__bar,
                   __global Kernel_ctx *__k_ctx, CL_Scheduler_ctx __s_ctx,
                   __local int *__scratchpad,
                   Restoration_ctx *__restoration_ctx) {

  int local_stop = 0;

  if (__restoration_ctx->target != 0) {
    local_stop = __restoration_ctx->local_stop;
  }
  while (
      __restoration_ctx->target !=
      UCHAR_MAX /* substitute for 'true', which can cause compiler hangs */) {
    switch (__restoration_ctx->target) {
    case 0:
      if (!(1)) {
        return;
      }

      // Get ids
      int tid = b_get_global_id(__bar, __k_ctx);
      int stride = b_get_global_size(__bar, __k_ctx);

      // Original application --- vector_assign --- start

      // The original kernels used an 'if' here. We need a 'for' loop
      for (int i = tid; i < num_rows; i += stride) {
        x[i] = y[i];
      }

      // Original application --- vector_assign --- end

      // Inter-workgroup barrier
      {
        Restoration_ctx __to_fork;
        __to_fork.target = 1;
        __to_fork.local_stop = local_stop;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 1:
      __restoration_ctx->target = 0;

      // Get ids
      tid = b_get_global_id(__bar, __k_ctx);
      stride = b_get_global_size(__bar, __k_ctx);

      // Original application --- spmv_min_dot_plus_kernel --- start

      // This is the right place to initialize stop variable as it is
      // the only barrier interval which doesn't use stop
      *stop = 0;

      // The original kernels used an 'if' here. We need a 'for' loop
      for (int it = tid; it < num_rows; it += stride) {

        // Get the start and end pointers
        int row_start = row[it];
        int row_end = row[it + 1];

        // Perform + for each pair of elements and a reduction with min
        int min = x[it];
        for (int j = row_start; j < row_end; j++) {
          if (data[j] + x[col[j]] < min)
            min = data[j] + x[col[j]];
        }
        y[it] = min;
      }

      // Original application --- spmv_min_dot_plus_kernel --- end

      // Inter-workgroup barrier
      {
        Restoration_ctx __to_fork;
        __to_fork.target = 2;
        __to_fork.local_stop = local_stop;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 2:
      __restoration_ctx->target = 0;

      tid = b_get_global_id(__bar, __k_ctx);
      stride = b_get_global_size(__bar, __k_ctx);

      // Original application --- vector_diff --- start

      // The original kernels used an 'if' here. We need a 'for' loop
      for (int i = tid; i < num_rows; i += stride) {
        if (y[i] != x[i])
          *stop = 1;
      }

      // Original application --- vector_diff --- end

      // Inter-workgroup barrier
      {
        Restoration_ctx __to_fork;
        __to_fork.target = 3;
        __to_fork.local_stop = local_stop;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 3:
      __restoration_ctx->target = 0;

      // Check terminating condition before continuing
      if (*stop == 0) {
        return;
      }
    }
  }
}
//

kernel void mega_kernel(__global int *A, const int A_row, const int A_col,
                        __global int *B, const int B_row, const int B_col,
                        __global int *C, __global atomic_int *counter,
                        __global atomic_int *hash, const int num_rows,
                        __global int *row, __global int *col,
                        __global int *data, __global int *x, __global int *y,
                        __global int *stop, __global IW_barrier *bar,
                        __global Discovery_ctx *d_ctx,
                        __global Kernel_ctx *non_persistent_kernel_ctx,
                        __global Kernel_ctx *persistent_kernel_ctx,
                        SCHEDULER_ARGS) {
#define NON_PERSISTENT_KERNEL                                                  \
  matmult(A, A_row, A_col, B, B_row, B_col, C, counter, hash,                  \
          non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  sssp_combined(num_rows, row, col, data, x, y, stop, bar,                     \
                persistent_kernel_ctx, s_ctx, scratchpad, &r_ctx_local)
#include "main_device_body.cl"
}
//