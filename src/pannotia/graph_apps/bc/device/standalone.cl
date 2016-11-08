#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

float atomic_add_float(__global float *const address, const float value) {

  uint oldval, newval, readback;

  *(float *)&oldval = *address;
  *(float *)&newval = (*(float *)&oldval + value);
  while ((readback = atomic_cmpxchg((__global uint *)address, oldval,
                                    newval)) != oldval) {
    oldval = readback;
    *(float *)&newval = (*(float *)&oldval + value);
  }
  return *(float *)&oldval;
}

// Here we have the mega-kernel code. That is, the kernels
// above combined and the host side loop combined on the GPU
int mega_bfs_kernel_func(__global Discovery_ctx *__d_ctx,
                         __global IW_barrier *__bar, __global int *row,
                         __global int *col, __global int *d,
                         __global float *rho, __global int *p,
                         __global int *stop1, __global int *stop2,
                         __global int *stop3, const int num_nodes,
                         const int num_edges, __global int *global_dist) {

  __global int *write_stop = stop1;
  __global int *read_stop = stop2;
  __global int *buff_stop = stop3;
  __global int *swap;

  // Get participating global id and the stride
  int tid = p_get_global_id(__d_ctx);
  int stride = p_get_global_size(__d_ctx);
  int local_dist = 0;

  while (1) {

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_nodes; i += stride) {
      if (d[i] == local_dist) {

        // Get the starting and ending pointers
        // of the neighbor list
        int start = row[i];
        int end;
        if (i + 1 < num_nodes)
          end = row[i + 1];
        else
          end = num_edges;

        // Navigate through the neighbor list
        for (int edge = start; edge < end; edge++) {
          int w = col[edge];
          if (d[w] < 0) {
            *write_stop = 1;

            // Traverse another layer
            d[w] = local_dist + 1;
          }

          // Transfer the rho value to the neighbor
          if (d[w] == (local_dist + 1)) {
            atomic_add_float(&rho[w], rho[i]);
          }
        }
      }
    }

    global_barrier_disc(__bar, __d_ctx);

    swap = read_stop;
    read_stop = write_stop;
    write_stop = buff_stop;
    buff_stop = swap;
    local_dist = local_dist + 1;

    // Inter-workgroup barrier
    global_barrier_disc(__bar, __d_ctx);

    // Trick for updating the termination variables using the
    // 'buff_stop' variable without extra barriers and avoiding
    // data-races.
    if (*read_stop == 0) {
      break;
    }
    *buff_stop = 0;
  }

  return local_dist;
}

void mega_backtrack_kernel_func(__global Discovery_ctx *__d_ctx,
                                __global IW_barrier *__bar, __global int *row,
                                __global int *col, __global int *d,
                                __global float *rho, __global float *sigma,
                                __global int *p, const int num_nodes,
                                const int num_edges, const int dist,
                                const int s, __global float *bc) {

  // Get global participating id and stride
  int tid = p_get_global_id(__d_ctx);
  int stride = p_get_global_size(__d_ctx);
  int local_dist = dist;

  while (local_dist > 0) {

    for (int i = tid; i < num_nodes; i += stride) {
      if (d[i] == local_dist - 1) {

        int start = row[i];
        int end;
        if (i + 1 < num_nodes)
          end = row[i + 1];
        else
          end = num_edges;

        // Get the starting and ending pointers
        // of the neighbor list in the reverse graph
        for (int edge = start; edge < end; edge++) {
          int w = col[edge];

          // Update the sigma value traversing back
          if (d[w] == local_dist - 2)
            atomic_add_float(&sigma[w],
                             rho[w] / rho[i] *
                                 ((1 * 10000) + sigma[i])); // Scaling by 10000
                                                            // to get correct
                                                            // results on Intel
        }

        // Update the BC value

        // Tyler: This looks like there might be a data-race here, but
        // the original authors assured me that there isn't.
        if (i != s)
          bc[i] = bc[i] + (sigma[i] / 10000); // Doing unscaling here
      }
    }
    local_dist = local_dist - 1;

    // Inter workgroup barrier
    global_barrier_disc(__bar, __d_ctx);
  }
}

__kernel void bc_combined(__global int *row,         // 0
                          __global int *col,         // 1
                          __global int *row_trans,   // 2
                          __global int *col_trans,   // 3
                          __global int *dist,        // 4
                          __global float *rho,       // 5
                          __global float *sigma,     // 6
                          __global int *p,           // 7
                          __global int *stop1,       // 8
                          __global int *stop2,       // 9
                          __global int *stop3,       // 10
                          __global int *global_dist, // 11
                          __global float *bc,        // 12
                          const int num_nodes,       // 13
                          const int num_edges, __global IW_barrier *__bar,
                          __global Discovery_ctx *__d_ctx, SCHEDULER_ARGS // 14
                          ) {

  // Original application --- clean_1d_array --- start

  __local int __scratchpad[2];
  DISCOVERY_PROTOCOL(__d_ctx, __scratchpad);
  INIT_SCHEDULER;
  int local_dist = -1;
  int s = 0;

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
    // for (int s = 0; s < num_nodes; s++) {
    if (!(s < num_nodes)) {
      break;
    }
    int tid = p_get_global_id(__d_ctx);
    int stride = p_get_global_size(__d_ctx);

    for (int i = tid; i < num_nodes; i += stride) {
      sigma[i] = 0;

      // If source vertex rho = 1, dist = 0
      if (i == s) {
        rho[i] = 1 * 10000; // Scaling by 10000
        dist[i] = 0;

      } else { // If other vertices rho = 0, dist = -1
        rho[i] = 0;
        dist[i] = -1;
      }
    }

    // Original application --- clean 1d_array --- end

    // No barrier required here because the two kernels
    // access disjoint memory.

    // Original application --- clean 2d_array --- start

    for (int i = tid; i < num_nodes * num_nodes; i += stride) {
      p[i] = 0;
    }

    // Original application --- clean 2d_array --- end

    // Inter workgroup barrier
    global_barrier_disc(__bar, __d_ctx);

    // Original application --- bfs_kernel --- start

    local_dist =
        mega_bfs_kernel_func(__d_ctx, __bar, row, col, dist, rho, p, stop1,
                             stop2, stop3, num_nodes, num_edges, global_dist);

    // Original application --- bfs_kernel --- end

    // Inter workgroup barrier
    global_barrier_disc(__bar, __d_ctx);

    // Original application --- backtrack_kernel --- start

    mega_backtrack_kernel_func(__d_ctx, __bar, row_trans, col_trans, dist, rho,
                               sigma, p, num_nodes, num_edges, local_dist, s,
                               bc);

    // Original application --- backtrack_kernel --- end
    // Inter workgroup barrier
    global_barrier_disc(__bar, __d_ctx);
    s = s + 1;
  }
  if (get_local_id(0) == 0) {
    atomic_fetch_sub_explicit(s_ctx.persistent_flag, 1, memory_order_acq_rel,
                              memory_scope_all_svm_devices);
  }
}
////