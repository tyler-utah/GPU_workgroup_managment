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

  barrier(CLK_GLOBAL_MEM_FENCE);

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
int mega_bfs_kernel_func(__global IW_barrier *__bar,
                         __global Kernel_ctx *__k_ctx, __local int *__sense,
                         __global int *row, __global int *col, __global int *d,
                         __global float *rho, __global int *p,
                         __global int *stop1, __global int *stop2,
                         __global int *stop3, const int num_nodes,
                         const int num_edges, __global int *global_dist) {

  __global int *write_stop = stop1;
  __global int *read_stop = stop2;
  __global int *buff_stop = stop3;
  __global int *swap;

  // Get participating global id and the stride
  int tid = b_get_global_id(__bar, __k_ctx);
  int stride = b_get_global_size(__bar, __k_ctx);
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

    global_barrier_robust_to_resizing(__bar, __sense, __k_ctx);

    swap = read_stop;
    read_stop = write_stop;
    write_stop = buff_stop;
    buff_stop = swap;
    local_dist = local_dist + 1;

    // Inter-workgroup barrier
    global_barrier_robust_to_resizing(__bar, __sense, __k_ctx);

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

void mega_backtrack_kernel_func(__global IW_barrier *__bar,
                                __global Kernel_ctx *__k_ctx,
                                __local int *__sense, __global int *row,
                                __global int *col, __global int *d,
                                __global float *rho, __global float *sigma,
                                __global int *p, const int num_nodes,
                                const int num_edges, const int dist,
                                const int s, __global float *bc) {

  // Get global participating id and stride
  int tid = b_get_global_id(__bar, __k_ctx);
  int stride = b_get_global_size(__bar, __k_ctx);
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
    global_barrier_robust_to_resizing(__bar, __sense, __k_ctx);
  }
}

void bc_combined(__global int *row,         // 0
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
                 __global Kernel_ctx *__k_ctx, CL_Scheduler_ctx __s_ctx,
                 __local int *__scratchpad, Restoration_ctx *__restoration_ctx,
                 __local int *__sense // 14
                 ) {

  // Original application --- clean_1d_array --- start

  int local_dist = -1;
  int s = 0;

  if (__restoration_ctx->target != 0) {
    *__sense = __restoration_ctx->__sense;
    local_dist = __restoration_ctx->local_dist;
    s = __restoration_ctx->s;
  }
  while (
      __restoration_ctx->target !=
      UCHAR_MAX /* substitute for 'true', which can cause compiler hangs */) {
    switch (__restoration_ctx->target) {
    case 0:
      if (!(1)) {
        return;
      }

      // for (int s = 0; s < num_nodes; s++) {
      if (!(s < num_nodes)) {
        return;
      }
      int tid = b_get_global_id(__bar, __k_ctx);
      int stride = b_get_global_size(__bar, __k_ctx);

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
      {
        Restoration_ctx __to_fork;
        __to_fork.target = 1;
        __to_fork.__sense = *__sense;
        __to_fork.local_dist = local_dist;
        __to_fork.s = s;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 1:
      __restoration_ctx->target = 0;

      // Original application --- bfs_kernel --- start

      local_dist = mega_bfs_kernel_func(__bar, __k_ctx, __sense, row, col, dist,
                                        rho, p, stop1, stop2, stop3, num_nodes,
                                        num_edges, global_dist);

      // Original application --- bfs_kernel --- end

      // Inter workgroup barrier
      {
        Restoration_ctx __to_fork;
        __to_fork.target = 2;
        __to_fork.__sense = *__sense;
        __to_fork.local_dist = local_dist;
        __to_fork.s = s;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 2:
      __restoration_ctx->target = 0;

      // Original application --- backtrack_kernel --- start

      mega_backtrack_kernel_func(__bar, __k_ctx, __sense, row_trans, col_trans,
                                 dist, rho, sigma, p, num_nodes, num_edges,
                                 local_dist, s, bc);

      // Original application --- backtrack_kernel --- end
      // Inter workgroup barrier
      {
        Restoration_ctx __to_fork;
        __to_fork.target = 3;
        __to_fork.__sense = *__sense;
        __to_fork.local_dist = local_dist;
        __to_fork.s = s;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 3:
      __restoration_ctx->target = 0;
      s = s + 1;
    }
  }
}
//

kernel void mega_kernel(__global int *A, const int A_row, const int A_col,
                        __global int *B, const int B_row, const int B_col,
                        __global int *C, __global atomic_int *counter,
                        __global atomic_int *hash, __global int *row, // 0
                        __global int *col,                            // 1
                        __global int *row_trans,                      // 2
                        __global int *col_trans,                      // 3
                        __global int *dist,                           // 4
                        __global float *rho,                          // 5
                        __global float *sigma,                        // 6
                        __global int *p,                              // 7
                        __global int *stop1,                          // 8
                        __global int *stop2,                          // 9
                        __global int *stop3,                          // 10
                        __global int *global_dist,                    // 11
                        __global float *bc,                           // 12
                        const int num_nodes,                          // 13
                        const int num_edges, __global IW_barrier *bar,
                        __global Discovery_ctx *d_ctx,
                        __global Kernel_ctx *non_persistent_kernel_ctx,
                        __global Kernel_ctx *persistent_kernel_ctx,
                        SCHEDULER_ARGS) {
#define NON_PERSISTENT_KERNEL                                                  \
  matmult(A, A_row, A_col, B, B_row, B_col, C, counter, hash,                  \
          non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  bc_combined(row, col, row_trans, col_trans, dist, rho, sigma, p, stop1,      \
              stop2, stop3, global_dist, bc, num_nodes, num_edges, bar,        \
              persistent_kernel_ctx, s_ctx, scratchpad, &r_ctx_local,          \
              &__sense)

  __local int __sense;
  __sense = 0;

#include "main_device_body.cl"
}
//