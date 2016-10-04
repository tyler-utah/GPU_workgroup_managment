
#include "../rt_common/cl_scheduler.h"
#include "kernel_ctx.cl"
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
    int local_flag = atomic_load_explicit(s_ctx.scheduler_flag, memory_order_acquire, memory_scope_all_svm_devices);
	
    // Routine to quit
    if (local_flag == DEVICE_TO_QUIT) {
      for(int i = 0; i < MAX_P_GROUPS; i++) {
		  
	    while (atomic_load_explicit(&(s_ctx.task_array[i]), memory_order_relaxed, memory_scope_device) !=  TASK_WAIT);
		
        atomic_store_explicit(&(s_ctx.task_array[i]), TASK_QUIT, memory_order_release, memory_scope_device);
		
		atomic_fetch_sub(s_ctx.available_workgroups, 1);
	  }
	  break;
	}

    // Routine to send tasks to graphics task
	if (local_flag == DEVICE_TO_TASK) {
	  int local_task_size = *(s_ctx.task_size);
	  graphics_kernel_ctx->num_groups = local_task_size;
	  atomic_store_explicit(&(graphics_kernel_ctx->completed), 0, memory_order_relaxed, memory_scope_device);
	  int lpg = *(s_ctx.participating_groups);

	  // Wait until we have all the groups we need. This will be the response time.
	  while (atomic_load_explicit(s_ctx.available_workgroups, memory_order_acquire, memory_scope_device) < local_task_size);

	  
	  
	  atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_GOT_GROUPS, memory_order_release, memory_scope_all_svm_devices);
      while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_relaxed, memory_scope_all_svm_devices) != DEVICE_TO_EXECUTE);

	  // Send the tasks to do the task. 
	  for(int i = 0; i < local_task_size; i++) {
		  
		// the index is lpg - 1 because we're already missing one group because of the scheduler.

		// Write the group id:
		graphics_kernel_ctx->group_ids[lpg - i] = i;
		
		
		while (atomic_load_explicit(&(s_ctx.task_array[lpg - i]), memory_order_relaxed, memory_scope_device) !=  TASK_WAIT);
	  
		// Set the task
	    atomic_store_explicit(&(s_ctx.task_array[lpg - i]), TASK_MULT, memory_order_release, memory_scope_device);
		
		atomic_fetch_sub(s_ctx.available_workgroups, 1);

	  }
	  
	  while (atomic_load_explicit(&(graphics_kernel_ctx->completed), memory_order_relaxed, memory_scope_device) != local_task_size);		  
	  
	  atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING, memory_order_release, memory_scope_all_svm_devices);
	}
	
	// Routine to send tasks to the persistent task
	if (local_flag == DEVICE_TO_PERSISTENT_TASK) {
	  
	  int local_task_size = *(s_ctx.task_size);
	  
	  persistent_kernel_ctx->num_groups = local_task_size;
	  
	  atomic_store_explicit(&(persistent_kernel_ctx->completed), 0, memory_order_relaxed, memory_scope_device);
	  
	  int lpg = *(s_ctx.participating_groups);

	  // Send the tasks to do the task. 
	  for(int i = 0; i < local_task_size; i++) {
		  
		// the index is i + 1 because we're already missing one group because of the scheduler.

		// Write the group id:
		persistent_kernel_ctx->group_ids[i + 1] = i;
	  
	  	while (atomic_load_explicit(&(s_ctx.task_array[i + 1]), memory_order_acquire, memory_scope_device) !=  TASK_WAIT);

	  
		// Set the task
	    atomic_store_explicit(&(s_ctx.task_array[i + 1]), TASK_PERSIST, memory_order_release, memory_scope_device);
		
		atomic_fetch_sub(s_ctx.available_workgroups, 1);
	  }
	  
	  atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING, memory_order_release, memory_scope_all_svm_devices);
	}
  }
}

#define SCHEDULER_ARGS __global atomic_int *scheduler_flag,      \
                       __global int *scheduler_groups,           \
					   __global atomic_int *task_array,          \
					   __global int *task_size,                  \
					   __global atomic_int *available_workgroups
					   
					   
#define INIT_SCHEDULER \
CL_Scheduler_ctx s_ctx;                              \
  s_ctx.scheduler_flag = scheduler_flag;             \
  s_ctx.participating_groups = scheduler_groups;     \
  s_ctx.task_array = task_array;                     \
  s_ctx.task_size = task_size;                       \
  s_ctx.available_workgroups = available_workgroups;  
  