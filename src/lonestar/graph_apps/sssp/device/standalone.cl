#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

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

uint processedge2(__global uint *dist, uint iteration, uint edge, uint src,
                  uint *dst,
                  // Graph
                  uint g_nnodes,

                  __global uint *g_edgessrcdst,

                  __global uint *g_psrc, __global uint *g_edgessrcwt) {

  *dst = g_edgessrcdst[edge];
  if (*dst >= g_nnodes)
    return 0;

  uint wt = g_edgessrcwt[edge];
  if (wt >= MYINFINITY)
    return 0;

  uint dstwt = dist[*dst];
  uint altdist = dist[src] + wt;

  if (altdist < dstwt) {
    atomic_min(&dist[*dst], altdist);
    return 1;
  }

  return 0;
}

// Maybe more ckill and cfork could be added in here?
uint processnode2(__global Discovery_ctx *__d_ctx, __global uint *dist,
                  __global int *in_wl, __global int *in_index,
                  __global int *out_wl, __global int *out_index,
                  __local int *gather_offsets, __local int *queue_index,
                  __local int *scan_arr, __local int *loc_tmp, __local int *src,
                  unsigned iteration,
                  // Graph
                  uint g_nnodes,

                  __global uint *g_edgessrcdst,

                  __global uint *g_psrc, __global uint *g_edgessrcwt

                  ) {

  const int SCRATCHSIZE = WGS;
  int nn;
  unsigned id = p_get_global_id(__d_ctx);
  int threads = p_get_global_size(__d_ctx);
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
        src[scratch_offset + i - done] = nn;
      }

      neighborsdone += i;
      scratch_offset += i;

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      int ncnt = 0;
      unsigned to_push = 0;

      if (get_local_id(0) < total_edges) {

        if (processedge2(dist, iteration, gather_offsets[get_local_id(0)],
                         src[get_local_id(0)], &to_push, g_nnodes,
                         g_edgessrcdst, g_psrc, g_edgessrcwt)) {
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

void drelax(__global Discovery_ctx *__d_ctx, __global uint *dist,
            __global int *in_wl, __global int *in_index, __global int *out_wl,
            __global int *out_index, __local int *gather_offsets,
            __local int *queue_index, __local int *scan_arr,
            __local int *loc_tmp, __local int *src, int iteration,

            // Graph
            uint g_nnodes,

            __global uint *g_edgessrcdst,

            __global uint *g_psrc, __global uint *g_edgessrcwt) {

  unsigned id = p_get_global_id(__d_ctx);

  if (iteration == 0) {

    if (id == 0) {
      int item = 0;
      wl_push(out_wl, out_index, item);
    }
    return;
  } else {
    processnode2(__d_ctx, dist, in_wl, in_index, out_wl, out_index,
                 gather_offsets, queue_index, scan_arr, loc_tmp, src, iteration,
                 g_nnodes, g_edgessrcdst, g_psrc, g_edgessrcwt);
  }
}

__kernel void drelax2(__global uint *dist,

                      // Input worklist
                      __global int *a_inwl_index, __global int *a_outwl_index,

                      // Output worklist
                      __global int *a_inwl_wl, __global int *a_outwl_wl,

                      // Graph
                      uint g_nnodes,

                      __global uint *g_edgessrcdst,

                      __global uint *g_psrc, __global uint *g_edgessrcwt,
                      __global IW_barrier *__bar,
                      __global Discovery_ctx *__d_ctx, SCHEDULER_ARGS) {

  __local int __scratchpad[2];
  DISCOVERY_PROTOCOL(__d_ctx, __scratchpad);
  INIT_SCHEDULER;
  int iteration = 0;
  __global int *in_wl = a_inwl_wl;
  __global int *out_wl = a_outwl_wl;
  __global int *in_index = a_inwl_index;
  __global int *out_index = a_outwl_index;

  __local int gather_offsets[WGS];
  __local int queue_index[1];
  __local int scan_arr[WGS];
  __local int loc_tmp[1];
  __local int src[WGS];

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

    drelax(__d_ctx, dist, in_wl, in_index, out_wl, out_index, gather_offsets,
           queue_index, scan_arr, loc_tmp, src, iteration, g_nnodes,
           g_edgessrcdst, g_psrc, g_edgessrcwt);

    global_barrier_disc(__bar, __d_ctx);

    __global int *tmp_wl = in_wl;
    __global int *tmp_index = in_index;
    in_wl = out_wl;
    in_index = out_index;
    out_wl = tmp_wl;
    out_index = tmp_index;

    *out_index = 0;

    iteration++;

    global_barrier_disc(__bar, __d_ctx);

    if (*(in_index) <= 0) {
      break;
    }
  }
  if (get_local_id(0) == 0) {
    atomic_fetch_sub_explicit(s_ctx.persistent_flag, 1, memory_order_acq_rel,
                              memory_scope_all_svm_devices);
  }
}
////