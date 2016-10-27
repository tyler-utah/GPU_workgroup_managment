#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

__global int __junk_global;

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

// mega_kernel: combines the color1 and color2 kernels using an
// inter-workgroup barrier and the discovery protocol
void color_combined(__global int *row,          // 0
                    __global int *col,          // 1
                    __global float *node_value, // 2
                    __global int *color_array,  // 3
                    __global int *stop1,        // 4
                    __global int *stop2,        // 5
                    __global float *max_d,      // 6
                    const int num_nodes,        // 7
                    const int num_edges, __global IW_barrier *__bar,
                    __global Kernel_ctx *__k_ctx, CL_Scheduler_ctx __s_ctx,
                    __local int *__scratchpad,
                    Restoration_ctx *__restoration_ctx) { // 8

  __global int *write_stop = stop1;
  __global int *read_stop = stop2;
  __global int *swap;
  int graph_color = 1;

  if (__restoration_ctx->target != 0) {
    write_stop = __restoration_ctx->write_stop;
    read_stop = __restoration_ctx->read_stop;
    swap = __restoration_ctx->swap;
    graph_color = __restoration_ctx->graph_color;
  }
  while (
      __restoration_ctx->target !=
      UCHAR_MAX /* substitute for 'true', which can cause compiler hangs */) {
    switch (__restoration_ctx->target) {
    case 0:
      if (!(1)) {
        return;
      }

      // Original application --- color --- start

      int tid = b_get_global_id(__bar, __k_ctx);
      int stride = b_get_global_size(__bar, __k_ctx);

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
      {
        Restoration_ctx __to_fork;
        __to_fork.target = 1;
        __to_fork.write_stop = write_stop;
        __to_fork.read_stop = read_stop;
        __to_fork.swap = swap;
        __to_fork.graph_color = graph_color;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 1:
      __restoration_ctx->target = 0;

      tid = b_get_global_id(__bar, __k_ctx);
      stride = b_get_global_size(__bar, __k_ctx);

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
        return;
      }

      graph_color = graph_color + 1;
      *write_stop = 0;

      // Original application --- color2 --- end

      // Inter-workgroup barrier
      {
        Restoration_ctx __to_fork;
        __to_fork.target = 2;
        __to_fork.write_stop = write_stop;
        __to_fork.read_stop = read_stop;
        __to_fork.swap = swap;
        __to_fork.graph_color = graph_color;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 2:
      __restoration_ctx->target = 0;
    }
  }
}
//

kernel void mega_kernel(__global int *buffer, int length,
                        __global atomic_int *result, __global int *row, // 0
                        __global int *col,                              // 1
                        __global float *node_value,                     // 2
                        __global int *color_array,                      // 3
                        __global int *stop1,                            // 4
                        __global int *stop2,                            // 5
                        __global float *max_d,                          // 6
                        const int num_nodes,                            // 7
                        const int num_edges, __global IW_barrier *bar,
                        __global Discovery_ctx *d_ctx,
                        __global Kernel_ctx *non_persistent_kernel_ctx,
                        __global Kernel_ctx *persistent_kernel_ctx,
                        SCHEDULER_ARGS) {
  __local int scratch[256];
#define NON_PERSISTENT_KERNEL                                                  \
  MY_reduce(buffer, length, result, scratch, non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  color_combined(row, col, node_value, color_array, stop1, stop2, max_d,       \
                 num_nodes, num_edges, bar, persistent_kernel_ctx, s_ctx,      \
                 scratchpad, &r_ctx_local)
#include "main_device_body.cl"
}
//