
#include "../rt_common/cl_scheduler.h"
#include "ocl_utility.cl"

int get_task(CL_Scheduler_ctx s_ctx, int group_id) {
  __local int task;
  if (get_local_id(0) == 0) {
    // Could be optimised to place a fence after the spin
    while (atomic_load_explicit(&(s_ctx.task_array[group_id]), memory_order_acquire, memory_scope_device) == TASK_WAIT);
	task = atomic_load_explicit(&(s_ctx.task_array[group_id]), memory_order_acquire, memory_scope_device);
  }
  BARRIER;
  return task;
}