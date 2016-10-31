#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

// This kernel is the naive version of the mega-kernel using
// the global barrier
__kernel void sssp_combined(const int num_rows, __global int *row,
                            __global int *col, __global int *data,
                            __global int *x, __global int *y,
                            __global int *stop, __global IW_barrier *__bar,
                            __global Discovery_ctx *__d_ctx, SCHEDULER_ARGS) {

  __local int __scratchpad[2];
  DISCOVERY_PROTOCOL(__d_ctx, __scratchpad);
  INIT_SCHEDULER;
  int local_stop = 0;

  if (p_get_group_id(__d_ctx) == 0) {
    if (get_local_id(0) == 0) {
      atomic_store_explicit(s_ctx.persistent_flag, __d_ctx->count,
                            memory_order_release, memory_scope_all_svm_devices);
      atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING,
                            memory_order_release, memory_scope_all_svm_devices);
      while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_acquire,
                                  memory_scope_all_svm_devices) !=
             DEVICE_TO_PERSISTENT_TASK)
        ;
    }
    BARRIER;
  }
  global_barrier_disc(__bar, __d_ctx);
  while (1) {

    // Get ids
    int tid = p_get_global_id(__d_ctx);
    int stride = p_get_global_size(__d_ctx);

    // Original application --- vector_assign --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_rows; i += stride) {
      x[i] = y[i];
    }

    // Original application --- vector_assign --- end

    // Inter-workgroup barrier
    global_barrier_disc(__bar, __d_ctx);

    // Get ids
    tid = p_get_global_id(__d_ctx);
    stride = p_get_global_size(__d_ctx);

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
    global_barrier_disc(__bar, __d_ctx);

    tid = p_get_global_id(__d_ctx);
    stride = p_get_global_size(__d_ctx);

    // Original application --- vector_diff --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_rows; i += stride) {
      if (y[i] != x[i])
        *stop = 1;
    }

    // Original application --- vector_diff --- end

    // Inter-workgroup barrier
    global_barrier_disc(__bar, __d_ctx);

    // Check terminating condition before continuing
    if (*stop == 0) {
      break;
    }
  }
  if (get_local_id(0) == 0) {
    atomic_fetch_sub_explicit(s_ctx.persistent_flag, 1, memory_order_acq_rel,
                              memory_scope_all_svm_devices);
  }
}
////