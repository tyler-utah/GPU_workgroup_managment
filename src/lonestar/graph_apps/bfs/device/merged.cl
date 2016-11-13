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

/** Breadth-first search -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Example breadth-first search application for demoing Galois system.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

// bfs portable barrier kernel code (using the discovery
// protocol). Ported from the GPU-Lonestar bfs-worklistc application
// by Tyler Sorensen (2016)

#define MYINFINITY 1000000000
#define WGS 128

void block_int_exclusive_sum_scan(__local int *s, __local int *tmp, int input,
                                  int *output, int *total_edges) {

  int lid = get_local_id(0);
  s[lid] = input;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    int sum = 0;
    for (int i = 0; i < get_local_size(0); i++) {
      int x = s[i];
      s[i] = sum;
      sum += x;
    }

    *tmp = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  *output = s[lid];
  *total_edges = *tmp;

  return;
}

int wl_pop_id(__global int *wl, __global int *index, int id, int *item) {

  if (id < *(index)) {
    *item = wl[id];
    return 1;
  }
  return 0;
}

int wl_push(__global int *wl, __global int *index, int ele) {

  int lindex = atomic_add(index, 1);

  // if (lindex >= *(wl->dnsize))
  //  return 0;

  wl[lindex] = ele;
  return 1;
}

int wl_push_1item(__global int *wl, __global int *index,
                  __local int *queue_index, __local int *scan_arr,
                  __local int *loc_tmp, int nitem, int item,
                  int threads_per_block) {

  int total_items = 0;
  int thread_data = nitem;

  block_int_exclusive_sum_scan(scan_arr, loc_tmp, thread_data, &thread_data,
                               &total_items);

  if (get_local_id(0) == 0) {
    *queue_index = atomic_add(index, total_items);
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (nitem == 1) {
    wl[*queue_index + thread_data] = item;
  }

  return total_items;
}

uint processedge2(__global uint *dist, uint iteration, uint edge, uint *dst,
                  // Graph
                  uint g_nnodes,

                  __global uint *g_edgessrcdst,

                  __global uint *g_psrc) {

  *dst = g_edgessrcdst[edge];
  if (*dst >= g_nnodes)
    return 0;

  uint wt = 1;
  if (wt >= MYINFINITY)
    return 0;

  if (dist[*dst] == MYINFINITY) {
    dist[*dst] = iteration;
    return MYINFINITY;
  }
  return 0;
}

uint processnode2(__global IW_barrier *__bar, __global Kernel_ctx *__k_ctx,
                  __global uint *dist, __global int *in_wl,
                  __global int *in_index, __global int *out_wl,
                  __global int *out_index, __local int *gather_offsets,
                  __local int *queue_index, __local int *scan_arr,
                  __local int *loc_tmp, unsigned iteration,
                  // Graph
                  uint g_nnodes,

                  __global uint *g_edgessrcdst,

                  __global uint *g_psrc) {

  const int SCRATCHSIZE = WGS;
  int nn;
  unsigned id = b_get_global_id(__bar, __k_ctx);
  int threads = b_get_global_size(__bar, __k_ctx);
  int total_inputs = (*(in_index) + threads - 1) / (threads);

  gather_offsets[get_local_id(0)] = 0;

  while (total_inputs-- > 0) {
    int neighborsize = 0;
    int neighboroffset = 0;
    int scratch_offset = 0;
    int total_edges = 0;

    if (wl_pop_id(in_wl, in_index, id, &nn)) {

      if (nn != -1) {
        // neighborsize = g_getOutDegree(graph, nn);
        neighborsize = g_psrc[nn + 1] - g_psrc[nn];
        neighboroffset = g_psrc[nn];
      }
    }

    // Is scratch offset the correct intermediate value?? The partial exclusive
    // sum.
    block_int_exclusive_sum_scan(scan_arr, loc_tmp, neighborsize,
                                 &scratch_offset, &total_edges);

    int done = 0;
    int neighborsdone = 0;

    while (total_edges > 0) {
      int i;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      for (i = 0; neighborsdone + i < neighborsize &&
                  (scratch_offset + i - done) < SCRATCHSIZE;
           i++) {
        gather_offsets[scratch_offset + i - done] =
            neighboroffset + neighborsdone + i;
      }

      neighborsdone += i;
      scratch_offset += i;

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      int ncnt = 0;
      unsigned to_push = 0;

      if (get_local_id(0) < total_edges) {

        if (processedge2(dist, iteration, gather_offsets[get_local_id(0)],
                         &to_push, g_nnodes, g_edgessrcdst, g_psrc)) {
          ncnt = 1;
        }
      }

      wl_push_1item(out_wl, out_index, queue_index, scan_arr, loc_tmp, ncnt,
                    (int)to_push, WGS);

      total_edges -= WGS;
      done += WGS;
    }

    id += threads;
  }
  return 0;
}

void drelax(__global IW_barrier *__bar, __global Kernel_ctx *__k_ctx,
            __global uint *dist, __global int *in_wl, __global int *in_index,
            __global int *out_wl, __global int *out_index,
            __local int *gather_offsets, __local int *queue_index,
            __local int *scan_arr, __local int *loc_tmp, int iteration,

            // Graph
            uint g_nnodes,

            __global uint *g_edgessrcdst,

            __global uint *g_psrc) {

  unsigned id = b_get_global_id(__bar, __k_ctx);

  if (iteration == 0) {

    if (id == 0) {
      int item = 0;
      wl_push(out_wl, out_index, item);
    }
    return;
  } else {
    processnode2(__bar, __k_ctx, dist, in_wl, in_index, out_wl, out_index,
                 gather_offsets, queue_index, scan_arr, loc_tmp, iteration,
                 g_nnodes, g_edgessrcdst, g_psrc);
  }
}

void drelax2(__global uint *dist,

             // Input worklist
             __global int *a_inwl_index, __global int *a_outwl_index,

             // Output worklist
             __global int *a_inwl_wl, __global int *a_outwl_wl,

             // Graph
             uint g_nnodes,

             __global uint *g_edgessrcdst,

             __global uint *g_psrc, __local int *gather_offsets,
             __local int *queue_index, __local int *scan_arr,
             __local int *loc_tmp, __global IW_barrier *__bar,
             __global Kernel_ctx *__k_ctx, CL_Scheduler_ctx __s_ctx,
             __local int *__scratchpad, Restoration_ctx *__restoration_ctx, __local int * __sense) {

  int iteration = 0;
  __global int *in_wl = a_inwl_wl;
  __global int *out_wl = a_outwl_wl;
  __global int *in_index = a_inwl_index;
  __global int *out_index = a_outwl_index;

  ;
  ;
  ;
  ;

  if (__restoration_ctx->target != 0) {
    iteration = __restoration_ctx->iteration;
    in_wl = __restoration_ctx->in_wl;
    out_wl = __restoration_ctx->out_wl;
    in_index = __restoration_ctx->in_index;
    out_index = __restoration_ctx->out_index;
	*__sense = __restoration_ctx->sense;
  }
  while (
      __restoration_ctx->target !=
      UCHAR_MAX /* substitute for 'true', which can cause compiler hangs */) {
    switch (__restoration_ctx->target) {
    case 0:
      if (!(1)) {
        return;
      }

      drelax(__bar, __k_ctx, dist, in_wl, in_index, out_wl, out_index,
             gather_offsets, queue_index, scan_arr, loc_tmp, iteration,
             g_nnodes, g_edgessrcdst, g_psrc);

      {
        Restoration_ctx __to_fork;
        __to_fork.target = 1;
        __to_fork.iteration = iteration;
        __to_fork.in_wl = in_wl;
        __to_fork.out_wl = out_wl;
        __to_fork.in_index = in_index;
        __to_fork.out_index = out_index;
		__to_fork.sense = *__sense;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 1:
      __restoration_ctx->target = 0;

      __global int *tmp_wl = in_wl;
      __global int *tmp_index = in_index;
      in_wl = out_wl;
      in_index = out_index;
      out_wl = tmp_wl;
      out_index = tmp_index;

      *out_index = 0;

      iteration++;

     
      global_barrier_robust_to_resizing(__bar, __sense, __k_ctx);


      if (*(in_index) <= 0) {
        return;
      }
    }
  }
}
//

kernel void mega_kernel(__global int *A, const int A_row, const int A_col,
                        __global int *B, const int B_row, const int B_col,
                        __global int *C, __global atomic_int *counter,
                        __global atomic_int *hash, __global uint *dist,

                        // Input worklist
                        __global int *a_inwl_index, __global int *a_outwl_index,

                        // Output worklist
                        __global int *a_inwl_wl, __global int *a_outwl_wl,

                        // Graph
                        uint g_nnodes,

                        __global uint *g_edgessrcdst,

                        __global uint *g_psrc, __global IW_barrier *bar,
                        __global Discovery_ctx *d_ctx,
                        __global Kernel_ctx *non_persistent_kernel_ctx,
                        __global Kernel_ctx *persistent_kernel_ctx,
                        SCHEDULER_ARGS) {
  __local int gather_offsets[WGS];
  __local int queue_index[1];
  __local int scan_arr[WGS];
  __local int loc_tmp[1];
  __local int __sense;
  __sense = 0;
#define NON_PERSISTENT_KERNEL                                                  \
  matmult(A, A_row, A_col, B, B_row, B_col, C, counter, hash,                  \
          non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  drelax2(dist, a_inwl_index, a_outwl_index, a_inwl_wl, a_outwl_wl, g_nnodes,  \
          g_edgessrcdst, g_psrc, gather_offsets, queue_index, scan_arr,        \
          loc_tmp, bar, persistent_kernel_ctx, s_ctx, scratchpad,              \
          &r_ctx_local, &__sense)
#include "main_device_body.cl"
}
//