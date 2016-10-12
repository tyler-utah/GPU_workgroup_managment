#pragma once

#include "../rt_common/cl_scheduler.h"
#include "kernel_ctx.cl"
#include "ocl_utility.cl"

int try_lock(atomic_int *m) {
	return !(atomic_exchange_explicit(m, 1, memory_order_relaxed, memory_scope_device));
}

void scheduler_lock(atomic_int *m) {
  while (atomic_exchange_explicit(m, 1, memory_order_relaxed, memory_scope_device) != 0);
  atomic_work_item_fence(FULL_FENCE, memory_order_acquire, memory_scope_device);
}

void scheduler_unlock(atomic_int *m) {
  atomic_store_explicit(m, 0, memory_order_release, memory_scope_device);
}

int scheduler_needs_workgroups(CL_Scheduler_ctx s_ctx) {
  if (atomic_load_explicit(s_ctx.groups_to_kill, memory_order_relaxed, memory_scope_device) > 0) 
	return 1;
  return 0;
}

// Don't move this! It needs the lock functions defined above.
#include "scheduler_1.cl"

int get_task(CL_Scheduler_ctx s_ctx, int group_id, __local int * scratchpad, Restoration_ctx *r_ctx) {
  if (get_local_id(0) == 0) {
    // Could be optimised to place a fence after the spin
	int tmp;
	while (true) {
	  tmp = atomic_load_explicit(&(s_ctx.task_array[group_id]), memory_order_relaxed, memory_scope_device);
	  if (tmp != TASK_WAIT) {
	    break;
	  }
	}
    atomic_work_item_fence(FULL_FENCE, memory_order_acquire, memory_scope_work_group);
	*scratchpad = tmp;
	*r_ctx = s_ctx.r_ctx_arr[group_id];
  }
  BARRIER;
  return *scratchpad;
}

// Called by a single scheduler thread. Records the available groups and signals to the host
// that it is ready for a task
void scheduler_init(CL_Scheduler_ctx s_ctx, __global Discovery_ctx *d_ctx, __global Kernel_ctx *graphics_kernel_ctx, __global Kernel_ctx *persistent_kernel_ctx) {
	 *(s_ctx.participating_groups) = d_ctx->count - 1;
	 graphics_kernel_ctx->d_ctx = d_ctx;
	 persistent_kernel_ctx->d_ctx = d_ctx;
     atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING, memory_order_release, memory_scope_all_svm_devices);
}

void scheduler_loop(CL_Scheduler_ctx s_ctx,
                    __global Discovery_ctx *d_ctx,
					__global Kernel_ctx *graphics_kernel_ctx,
					__global Kernel_ctx *persistent_kernel_ctx) {
	
  // Scheduler loop
  while (true) {
    int local_flag = atomic_load_explicit(s_ctx.scheduler_flag, memory_order_relaxed, memory_scope_all_svm_devices);
	
    // Routine to quit (same for all schedulers)
    if (local_flag == DEVICE_TO_QUIT) {
		
      atomic_work_item_fence(FULL_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
	  
	  
      for(int i = 0; i < MAX_P_GROUPS; i++) {
		  
	    while (atomic_load_explicit(&(s_ctx.task_array[i]), memory_order_relaxed, memory_scope_device) !=  TASK_WAIT);
        atomic_store_explicit(&(s_ctx.task_array[i]), TASK_QUIT, memory_order_release, memory_scope_device);	
		atomic_fetch_sub(s_ctx.available_workgroups, 1);
	  }
	  break;
	}

    // Routine to send tasks to graphics task
	if (local_flag == DEVICE_TO_TASK) {
      atomic_work_item_fence(FULL_FENCE, memory_order_acquire, memory_scope_all_svm_devices);

	  int local_task_size = *(s_ctx.task_size);
	  atomic_store_explicit(&(graphics_kernel_ctx->num_groups), local_task_size, memory_order_relaxed, memory_scope_device);
	  atomic_store_explicit(&(graphics_kernel_ctx->executing_groups), local_task_size, memory_order_relaxed, memory_scope_device);
	  
	  scheduler_lock(s_ctx.pool_lock);
	  
	  // Needs to be atomic.
	  int to_kill = local_task_size - atomic_load_explicit(s_ctx.available_workgroups, memory_order_relaxed, memory_scope_device);
	  atomic_store_explicit(s_ctx.groups_to_kill, to_kill, memory_order_relaxed, memory_scope_device);
	  
	  // Wait until we have all the groups we need. This will be the response time.
	  while (atomic_load_explicit(s_ctx.available_workgroups, memory_order_relaxed, memory_scope_device) < local_task_size);

	  atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_GOT_GROUPS, memory_order_relaxed, memory_scope_all_svm_devices);
	  
      while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_relaxed, memory_scope_all_svm_devices) != DEVICE_TO_EXECUTE);
	  
	  scheduler_assign_tasks_graphics(s_ctx, graphics_kernel_ctx);
	  
	  scheduler_unlock(s_ctx.pool_lock);
	  
	  while (atomic_load_explicit(&(graphics_kernel_ctx->executing_groups), memory_order_relaxed, memory_scope_device) != 0);		  
	  
	  atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING, memory_order_release, memory_scope_all_svm_devices);
	}
	
	// Routine to send tasks to the persistent task
	// NOTE: THIS IS NOT SET UP TO WAIT FOR TASKS. IT PROBABLY SHOULD BE
	if (local_flag == DEVICE_TO_PERSISTENT_TASK) {
		
      atomic_work_item_fence(FULL_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
	  
	  int local_task_size = *(s_ctx.task_size);
	  
	  atomic_store_explicit(&(persistent_kernel_ctx->num_groups), local_task_size, memory_order_relaxed, memory_scope_device);
	  
	  //persistent_kernel_ctx->num_groups = local_task_size;
	  
	  atomic_store_explicit(&(persistent_kernel_ctx->executing_groups), local_task_size, memory_order_relaxed, memory_scope_device);
	  
	  int lpg = *(s_ctx.participating_groups);
	  
  	  scheduler_lock(s_ctx.pool_lock);
  
	  // Wait until we have all the groups we need. This will be the response time.
	  while (atomic_load_explicit(s_ctx.available_workgroups, memory_order_relaxed, memory_scope_device) < local_task_size);
	  
	  atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_GOT_GROUPS, memory_order_relaxed, memory_scope_all_svm_devices);
	  
      while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_relaxed, memory_scope_all_svm_devices) != DEVICE_TO_EXECUTE);

	  scheduler_assign_tasks_persistent(s_ctx, persistent_kernel_ctx);
	  
	  scheduler_unlock(s_ctx.pool_lock);
	  
	  atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING, memory_order_release, memory_scope_all_svm_devices);
	}
  }
}

#define SCHEDULER_ARGS __global atomic_int *scheduler_flag,       \
                       __global int *scheduler_groups,            \
					   __global atomic_int *task_array,           \
					   __global int *task_size,                   \
					   __global atomic_int *available_workgroups, \
					   __global atomic_int *pool_lock,            \
					   __global atomic_int *groups_to_kill,       \
					   __global atomic_int *persistent_flag,      \
					   __global Restoration_ctx *r_ctx_arr
					   
					   
#define INIT_SCHEDULER \
CL_Scheduler_ctx s_ctx;                              \
  s_ctx.scheduler_flag = scheduler_flag;             \
  s_ctx.participating_groups = scheduler_groups;     \
  s_ctx.task_array = task_array;                     \
  s_ctx.task_size = task_size;                       \
  s_ctx.available_workgroups = available_workgroups; \
  s_ctx.pool_lock = pool_lock;                       \
  s_ctx.groups_to_kill = groups_to_kill;             \
  s_ctx.persistent_flag = persistent_flag;           \
  s_ctx.r_ctx_arr = r_ctx_arr
  

  