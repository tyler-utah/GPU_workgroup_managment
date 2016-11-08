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

kernel void mega_kernel(__global int *A, const int A_row, const int A_col,
                        __global int *B, const int B_row, const int B_col,
                        __global int *C, __global atomic_int *counter,
                        __global atomic_int *hash, __global int *row, // 0
                        __global int *col,                            // 1
                        __global float *node_value,                   // 2
                        __global int *color_array,                    // 3
                        __global int *stop1,                          // 4
                        __global int *stop2,                          // 5
                        __global float *max_d,                        // 6
                        const int num_nodes,                          // 7
                        const int num_edges, __global IW_barrier *bar,
                        __global Discovery_ctx *d_ctx,
                        __global Kernel_ctx *non_persistent_kernel_ctx,
                        __global Kernel_ctx *persistent_kernel_ctx,
                        SCHEDULER_ARGS) {
#define NON_PERSISTENT_KERNEL                                                  \
  matmult(A, A_row, A_col, B, B_row, B_col, C, counter, hash,                  \
          non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  color_combined(row, col, node_value, color_array, stop1, stop2, max_d,       \
                 num_nodes, num_edges, bar, persistent_kernel_ctx, s_ctx,      \
                 scratchpad, &r_ctx_local)
#include "main_device_body.cl"
}
//