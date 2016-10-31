#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

// Simple min reduce from:
// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
void MY_reduce(__global int *buffer, int length, __global atomic_int *result,
               __local int *scratch, __global Kernel_ctx *__k_ctx) {

  ;
  int gid = k_get_global_id(__k_ctx);
  int local_index = get_local_id(0);
  int stride = k_get_global_size(__k_ctx);

  for (int global_index = gid; global_index < length; global_index += stride) {
    // Load data into local memory
    if (global_index < length) {
      scratch[local_index] = buffer[global_index];
    } else {
      // Infinity is the identity element for the min operation
      scratch[local_index] = INT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < get_local_size(0); offset <<= 1) {
      int mask = (offset << 1) - 1;
      if ((local_index & mask) == 0) {
        int other = scratch[local_index + offset];
        int mine = scratch[local_index];
        scratch[local_index] = (mine < other) ? mine : other;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Putting an atomic here so we can get a global reduction
    if (local_index == 0) {
      atomic_fetch_min((result), scratch[0]);
    }
  }
}
//

__global int __junk_global;

// This kernel is the naive version of the mega-kernel using
// the global barrier
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

kernel void mega_kernel(__global int *buffer, int length,
                        __global atomic_int *result, const int num_rows,
                        __global int *row, __global int *col,
                        __global int *data, __global int *x, __global int *y,
                        __global int *stop, __global IW_barrier *bar,
                        __global Discovery_ctx *d_ctx,
                        __global Kernel_ctx *non_persistent_kernel_ctx,
                        __global Kernel_ctx *persistent_kernel_ctx,
                        SCHEDULER_ARGS) {
  __local int scratch[256];
#define NON_PERSISTENT_KERNEL                                                  \
  MY_reduce(buffer, length, result, scratch, non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  sssp_combined(num_rows, row, col, data, x, y, stop, bar,                     \
                persistent_kernel_ctx, s_ctx, scratchpad, &r_ctx_local)
#include "main_device_body.cl"
}
//