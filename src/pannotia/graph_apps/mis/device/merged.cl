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

#define BIGNUM 99999999

// mega_kernel: combines the mis1, mis2, and mis3 kernels using an
// inter-workgroup barrier and the discovery protocol
void mis_combined(__global int *row, __global int *col,
                  __global float *node_value, __global int *s_array,
                  __global int *c_array, __global int *cu_array,
                  __global float *min_array, __global int *stop, int num_nodes,
                  int num_edges, __global IW_barrier *__bar,
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

      // Original application --- mis1 --- start
      // Get participating global id and the stride
      int tid_start = b_get_global_id(__bar, __k_ctx);
      int stride = b_get_global_size(__bar, __k_ctx);

      // The original kernels used an 'if' here. We need a 'for' loop
      for (int tid = tid_start; tid < num_nodes; tid += stride) {

        // If the vertex is not processed
        if (c_array[tid] == -1) {
          *stop = 1;

          // Get the start and end pointers
          int start = row[tid];
          int end;
          if (tid + 1 < num_nodes)
            end = row[tid + 1];
          else
            end = num_edges;

          // Navigate the neighbor list and find the min
          float min = BIGNUM;
          for (int edge = start; edge < end; edge++) {
            if (c_array[col[edge]] == -1) {
              if (node_value[col[edge]] < min)
                min = node_value[col[edge]];
            }
          }
          min_array[tid] = min;
        }
      }

      // Original application --- mis1 --- end

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
      local_stop = *stop;
      tid_start = b_get_global_id(__bar, __k_ctx);
      stride = b_get_global_size(__bar, __k_ctx);

      // Original application --- mis2 --- start

      // The original kernels used an 'if' here. We need a 'for' loop
      for (int tid = tid_start; tid < num_nodes; tid += stride) {

        if (node_value[tid] < min_array[tid] && c_array[tid] == -1) {

          // -1 : not processed
          // -2 : inactive
          //  2 : independent set put the item into the independent set
          s_array[tid] = 2;

          // Get the start and end pointers
          int start = row[tid];
          int end;

          if (tid + 1 < num_nodes)
            end = row[tid + 1];
          else
            end = num_edges;

          // Set the status to inactive
          c_array[tid] = -2;

          // Mark all the neighnors inactive
          for (int edge = start; edge < end; edge++) {
            if (c_array[col[edge]] == -1) {

              // Use status update array to avoid race
              cu_array[col[edge]] = -2;
            }
          }
        }
      }

      // Original application --- mis2 --- end

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
      tid_start = b_get_global_id(__bar, __k_ctx);
      stride = b_get_global_size(__bar, __k_ctx);

      if (local_stop == 0) {
        return;
      }
      *stop = 0;

      // Original application --- mis3 --- start

      // The original kernels used an 'if' here. We need a 'for' loop
      for (int tid = tid_start; tid < num_nodes; tid += stride) {
        if (cu_array[tid] == -2) {
          c_array[tid] = cu_array[tid];
        }
      }

      // Original application --- mis3 --- end

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
    }
  }
}
//

kernel void mega_kernel(__global int *buffer, int length,
                        __global atomic_int *result, __global int *row,
                        __global int *col, __global float *node_value,
                        __global int *s_array, __global int *c_array,
                        __global int *cu_array, __global float *min_array,
                        __global int *stop, int num_nodes, int num_edges,
                        __global IW_barrier *bar, __global Discovery_ctx *d_ctx,
                        __global Kernel_ctx *non_persistent_kernel_ctx,
                        __global Kernel_ctx *persistent_kernel_ctx,
                        SCHEDULER_ARGS) {
  __local int scratch[256];
#define NON_PERSISTENT_KERNEL                                                  \
  MY_reduce(buffer, length, result, scratch, non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  mis_combined(row, col, node_value, s_array, c_array, cu_array, min_array,    \
               stop, num_nodes, num_edges, bar, persistent_kernel_ctx, s_ctx,  \
               scratchpad, &r_ctx_local)
#include "main_device_body.cl"
}
//