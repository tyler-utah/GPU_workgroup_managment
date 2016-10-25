#pragma once

#include "kernel_ctx.cl"
#include "cl_scheduler.cl"

int query_workgroups_to_kill(CL_Scheduler_ctx s_ctx) {
	return atomic_load_explicit(s_ctx.groups_to_kill, memory_order_special_relax_acquire, memory_scope_device);
}

int offer_fork(__global Kernel_ctx * k_ctx, CL_Scheduler_ctx s_ctx, __local int *scratchpad, Restoration_ctx *r_ctx, int *former_groups, __global int *update) {
	
  if (get_local_id(0) == 0) {
	int h = *scratchpad;
	int pre_groups = k_get_num_groups(k_ctx);
    scratchpad[0] = pre_groups;
	scratchpad[1] = pre_groups;
	*update = pre_groups;

	if (atomic_load_explicit(s_ctx.available_workgroups, memory_order_special_relax_acquire, memory_scope_device) > 0) {
      if (try_lock(s_ctx.pool_lock)) {
		
        atomic_work_item_fence(FULL_FENCE, memory_order_acquire, memory_scope_device);

		int groups = k_get_num_groups(k_ctx);
		int snapshot = atomic_load_explicit(s_ctx.available_workgroups, memory_order_relaxed, memory_scope_device);
		*update = groups + snapshot;
		
	    for (int i = groups; i < groups+snapshot; i++) {
			
		  // Write the group id:
          k_ctx->group_ids[i + 1] = i;
		  while (atomic_load_explicit(&(s_ctx.task_array[i + 1]), memory_order_special_relax_acquire, memory_scope_device) !=  TASK_WAIT);
		  s_ctx.r_ctx_arr[i + 1] = *r_ctx;
		 
		  
		  // Set the task
          atomic_store_explicit(&(s_ctx.task_array[i + 1]), TASK_PERSIST, memory_order_release, memory_scope_device);
          
		}

		atomic_fetch_add_explicit(&(k_ctx->num_groups), snapshot, memory_order_relaxed, memory_scope_device);
		atomic_fetch_add_explicit(&(k_ctx->executing_groups), snapshot, memory_order_relaxed, memory_scope_device);
		atomic_fetch_add_explicit(s_ctx.persistent_flag, snapshot, memory_order_release, memory_scope_all_svm_devices);

		atomic_fetch_sub_explicit(s_ctx.available_workgroups, snapshot, memory_order_relaxed, memory_scope_device);

	    scheduler_unlock(s_ctx.pool_lock);
		scratchpad[0] = snapshot + groups;
		scratchpad[1] = groups;
	  } 
	}
  }
  
  BARRIER;
  *former_groups = scratchpad[1];
  return scratchpad[0];
}

int __offer_kill(__global Kernel_ctx * k_ctx, CL_Scheduler_ctx s_ctx, __local int *scratchpad, const int group_id) {

  if (get_local_id(0) == 0) {
    *scratchpad = 0;
    
	// This is essentially our lock
    if (group_id == k_get_num_groups(k_ctx) - 1) {
      atomic_work_item_fence(FULL_FENCE, memory_order_acquire, memory_scope_device);
	  int to_kill = atomic_load_explicit(s_ctx.groups_to_kill, memory_order_relaxed, memory_scope_device);
	  
	  if (to_kill > 0) {
		atomic_store_explicit(s_ctx.groups_to_kill, to_kill - 1, memory_order_relaxed, memory_scope_device);

		// This "releases" the lock
	    atomic_fetch_sub_explicit(&(k_ctx->num_groups), 1, memory_order_release, memory_scope_device);
		*scratchpad = -1;
	  }
	}
  }
  
  BARRIER;
  return *scratchpad;
}

void scheduler_assign_tasks_graphics(CL_Scheduler_ctx s_ctx, __global Kernel_ctx * graphics_kernel_ctx) {
	
  int local_task_size = *(s_ctx.task_size);
  int lpg = *(s_ctx.participating_groups);
  
  // Send the tasks to do the task. 
  for(int i = 0; i < local_task_size; i++) {
		
    // the index is lpg - 1 because we're already missing one group because of the scheduler.

    // Write the group id:
    graphics_kernel_ctx->group_ids[lpg - i] = i;

	// Two phase check because available workgroups and task_array may be out of sync
	while (atomic_load_explicit(&(s_ctx.task_array[lpg - i]), memory_order_special_relax_acquire, memory_scope_device) !=  TASK_WAIT);
	  
    // Set the task
	atomic_store_explicit(&(s_ctx.task_array[lpg - i]), TASK_MULT, memory_order_release, memory_scope_device);
		
    atomic_fetch_sub_explicit(s_ctx.available_workgroups, 1, memory_order_relaxed, memory_scope_device);

  }
}

void scheduler_assign_tasks_persistent(CL_Scheduler_ctx s_ctx, __global Kernel_ctx * persistent_kernel_ctx) {
	
  int local_task_size = *(s_ctx.task_size);
  int lpg = *(s_ctx.participating_groups);
  

  
  // Send the tasks to do the task. 
  for(int i = 0; i < local_task_size; i++) {
	  
	    


    // the index is i + 1 because we're already missing one group because of the scheduler.

    // Write the group id:
    persistent_kernel_ctx->group_ids[i + 1] = i;
	  
    while (atomic_load_explicit(&(s_ctx.task_array[i + 1]), memory_order_special_relax_acquire, memory_scope_device) !=  TASK_WAIT);
	
	// Set the restoration ctx
	s_ctx.r_ctx_arr[i+1].target = 0;

    // Set the task
    atomic_store_explicit(&(s_ctx.task_array[i + 1]), TASK_PERSIST, memory_order_release, memory_scope_device);
		
    atomic_fetch_sub_explicit(s_ctx.available_workgroups, 1, memory_order_relaxed, memory_scope_device);
  }
}

#define offer_kill_barrier(kernel_ctx, s_ctx, scratchpad, id) if (__offer_kill(kernel_ctx, s_ctx, scratchpad, id) == -1) {return -1;}
#define offer_kill(kernel_ctx, s_ctx, scratchpad, id) if (__offer_kill(kernel_ctx, s_ctx, scratchpad, id) == -1) {return;}