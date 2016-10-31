#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

__kernel void os_freq(__global const uint *num_iterations,
                      __global IW_barrier *__bar,
                      __global Discovery_ctx *__d_ctx
                      /* , SCHEDULER_ARGS */)
{
  __local int __scratchpad[2];

  DISCOVERY_PROTOCOL(__d_ctx, __scratchpad);
  /* INIT_SCHEDULER; */
  /* int dummy; */

  /* if (p_get_group_id(__d_ctx) == 0) { */
  /*   if (get_local_id(0) == 0) { */
  /*     atomic_store_explicit(s_ctx.persistent_flag, __d_ctx->count, */
  /*                           memory_order_release, memory_scope_all_svm_devices); */
  /*     atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING, */
  /*                           memory_order_release, memory_scope_all_svm_devices); */
  /*     while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_acquire, */
  /*                                 memory_scope_all_svm_devices) != */
  /*            DEVICE_TO_PERSISTENT_TASK) */
  /*       ; */
  /*   } */
  /*   BARRIER; */
  /* } */
  /* global_barrier_disc(__bar, __d_ctx); */

  while (1) {
    for (uint i = 0; i < *num_iterations; i++) {
      for (int j = 0; j < 10; j++) {
        global_barrier_disc(__bar, __d_ctx);
      }
    }
    break;
  }
  /* if (get_local_id(0) == 0) { */
  /*   atomic_fetch_sub_explicit(s_ctx.persistent_flag, 1, memory_order_acq_rel, */
  /*                             memory_scope_all_svm_devices); */
  /* } */
}
//
