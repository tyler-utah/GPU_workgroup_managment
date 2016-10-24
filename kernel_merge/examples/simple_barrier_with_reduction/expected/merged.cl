#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

void MY_reduce(int length, __global int *buffer, __global atomic_int *result,
               __local int *scratch, __global Kernel_ctx *__k_ctx) {

  ;
  int gid = k_get_global_id(__k_ctx);
  int local_index = get_local_id(0);
  int stride = k_get_global_size(__k_ctx);

  for (int global_index = gid; global_index < length; global_index += stride) {
    if (global_index < length) {
      scratch[local_index] = buffer[global_index];
    } else {
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
    if (local_index == 0) {
      atomic_fetch_min((result), scratch[0]);
    }
  }
}

void simple_barrier(__global IW_barrier *__bar, __global Kernel_ctx *__k_ctx,
                    CL_Scheduler_ctx __s_ctx, __local int *__scratchpad,
                    Restoration_ctx *__restoration_ctx) {
  int i = 0;
  if (__restoration_ctx->target != 0) {
    i = __restoration_ctx->i;
  }
  while (true) {
    switch (__restoration_ctx->target) {
    case 0:
      if (!(true)) {
        return;
      }

      i++;
      {
        Restoration_ctx __to_fork;
        __to_fork.target = 1;
        __to_fork.i = i;
        global_barrier_resize(__bar, __k_ctx, __s_ctx, __scratchpad,
                              &__to_fork);
      }
    case 1:
      __restoration_ctx->target = 0;
      if (i == 100000) {
        return;
      }
    }
  }
}

kernel void mega_kernel(int length, __global int *buffer,
                        __global atomic_int *result, __global IW_barrier *bar,
                        __global Discovery_ctx *d_ctx,
                        __global Kernel_ctx *non_persistent_kernel_ctx,
                        __global Kernel_ctx *persistent_kernel_ctx,
                        SCHEDULER_ARGS) {
  __local int scratch[256];
#define NON_PERSISTENT_KERNEL                                                  \
  MY_reduce(length, buffer, result, scratch, non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  simple_barrier(bar, persistent_kernel_ctx, s_ctx, scratchpad, &r_ctx_local)
#include "main_device_body.cl"
}
//