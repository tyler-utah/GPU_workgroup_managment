#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

// mega_kernel: combines the color1 and color2 kernels using an
// inter-workgroup barrier and the discovery protocol
__kernel void color_combined(__global int *row,          // 0
                             __global int *col,          // 1
                             __global float *node_value, // 2
                             __global int *color_array,  // 3
                             __global int *stop1,        // 4
                             __global int *stop2,        // 5
                             __global float *max_d,      // 6
                             const int num_nodes,        // 7
                             const int num_edges, __global IW_barrier *__bar,
                             __global Discovery_ctx *__d_ctx,
                             SCHEDULER_ARGS) { // 8

  __local int __scratchpad[2];
  DISCOVERY_PROTOCOL(__d_ctx, __scratchpad);
  INIT_SCHEDULER;
  __global int *write_stop = stop1;
  __global int *read_stop = stop2;
  __global int *swap;
  int graph_color = 1;

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

    // Original application --- color --- start

    int tid = p_get_global_id(__d_ctx);
    int stride = p_get_global_size(__d_ctx);

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_nodes; i += stride) {

      // If the vertex is still not colored
      if (color_array[i] == -1) {

        // Get the start and end pointer of the neighbor list
        int start = row[i];
        int end;
        if (i + 1 < num_nodes)
          end = row[i + 1];
        else
          end = num_edges;

        float maximum = -1;

        // Navigate the neighbor list
        for (int edge = start; edge < end; edge++) {

          // Determine if the vertex value is the maximum in the neighborhood
          if (color_array[col[edge]] == -1 && start != end - 1) {
            *write_stop = 1;
            if (node_value[col[edge]] > maximum)
              maximum = node_value[col[edge]];
          }
        }
        // Assign maximum the max array
        max_d[i] = maximum;
      }
    }

    // Two terminating variables allow us to only use 1
    // inter-workgroup barrier and still avoid a data-race
    swap = read_stop;
    read_stop = write_stop;
    write_stop = swap;

    // Original application --- color --- end

    // Inter-workgroup barrier
    global_barrier_disc(__bar, __d_ctx);

    tid = p_get_global_id(__d_ctx);
    stride = p_get_global_size(__d_ctx);

    // Original application --- color2 --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_nodes; i += stride) {

      // If the vertex is still not colored
      if (color_array[i] == -1) {
        if (node_value[i] > max_d[i])

          // Assign a color
          color_array[i] = graph_color;
      }
    }

    if (*read_stop == 0) {
      break;
    }

    graph_color = graph_color + 1;
    *write_stop = 0;

    // Original application --- color2 --- end

    // Inter-workgroup barrier
    global_barrier_disc(__bar, __d_ctx);
  }
  if (get_local_id(0) == 0) {
    atomic_fetch_sub_explicit(s_ctx.persistent_flag, 1, memory_order_acq_rel,
                              memory_scope_all_svm_devices);
  }
}
////