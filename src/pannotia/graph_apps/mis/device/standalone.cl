#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

#define BIGNUM 99999999

// mega_kernel: combines the mis1, mis2, and mis3 kernels using an
// inter-workgroup barrier and the discovery protocol
__kernel void mis_combined(__global int *row, __global int *col,
                           __global float *node_value, __global int *s_array,
                           __global int *c_array, __global int *cu_array,
                           __global float *min_array, __global int *stop,
                           int num_nodes, int num_edges,
                           __global IW_barrier *__bar,
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

    // Original application --- mis1 --- start
    // Get participating global id and the stride
    int tid_start = p_get_global_id(__d_ctx);
    int stride = p_get_global_size(__d_ctx);

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
    global_barrier_disc(__bar, __d_ctx);
    local_stop = *stop;
    tid_start = p_get_global_id(__d_ctx);
    stride = p_get_global_size(__d_ctx);

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
    global_barrier_disc(__bar, __d_ctx);
    tid_start = p_get_global_id(__d_ctx);
    stride = p_get_global_size(__d_ctx);

    if (local_stop == 0) {
      break;
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
    global_barrier_disc(__bar, __d_ctx);
  }
  if (get_local_id(0) == 0) {
    atomic_fetch_sub_explicit(s_ctx.persistent_flag, 1, memory_order_acq_rel,
                              memory_scope_all_svm_devices);
  }
}
////