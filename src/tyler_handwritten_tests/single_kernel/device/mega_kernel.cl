
#include "discovery.cl"
#include "cl_scheduler.cl"
#include "kernel_ctx.cl"

// Simple min reduce from:
// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
void MY_reduce(
            __global int* buffer,
            int length,
            __global atomic_int* result,
			int group_id,
			int num_groups) {
				
  __local int scratch[256];
  int gid = group_id * get_local_size(0) + get_local_id(0);
  int local_index = get_local_id(0);
  int stride = num_groups * get_local_size(0);
  
  for (int global_index = gid; global_index < length; global_index += stride) {
    // Load data into local memory
    if (global_index < length) {
      scratch[local_index] = buffer[global_index];
    } else {
      // Infinity is the identity element for the min operation
      scratch[local_index] = INT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = 1;
        offset < get_local_size(0);
        offset <<= 1) {
      int mask = (offset << 1) - 1;
      if ((local_index & mask) == 0) {
        int other = scratch[local_index + offset];
        int mine = scratch[local_index];
        scratch[local_index] = (mine < other) ? mine : other;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
	// Putting an atomic here so we can get a global reduction
    if (local_index == 0) {
	   //atomic_store(result, length);
      //result[get_group_id(0)] = scratch[0];
	  atomic_fetch_min((result), scratch[0]);
    }
  }
}

__kernel void mega_kernel(__global Discovery_ctx *d_ctx,
                          __global Kernel_ctx *kernel_ctx,
						  
						  //Kernel args
						  __global int * kernel_buffer,
						  __global atomic_int * kernel_result,
						  int kernel_length,
						  
                         //Many scheduler args are uvm primitives and need
						 //to be passed individually. 
                         __global atomic_int *scheduler_flag,
						 __global int *scheduler_groups,
						 __global atomic_int *task_array,
						 __global int *task_size){
							 

  DISCOVERY_PROTOCOL(d_ctx);
  
  // Scheduler init
  CL_Scheduler_ctx s_ctx;
  s_ctx.scheduler_flag = scheduler_flag;
  s_ctx.participating_groups = scheduler_groups;
  s_ctx.task_array = task_array; 
  s_ctx.task_size = task_size;  

  int group_id = p_get_group_id(d_ctx); 

  // Scheduler workgroup
  if (group_id == 0) {
    if (get_local_id(0) == 0) {
	
	  *(s_ctx.participating_groups) = d_ctx->count - 1;
      atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING, memory_order_release, memory_scope_all_svm_devices);
	  
	  // Scheduler loop
	  while (true) {
	    int local_flag = atomic_load_explicit(s_ctx.scheduler_flag, memory_order_acquire, memory_scope_all_svm_devices);
		
		if (local_flag == DEVICE_TO_QUIT) {
		  for(int i = 0; i < MAX_P_GROUPS; i++) {
			atomic_store_explicit(&(s_ctx.task_array[i]), TASK_QUIT, memory_order_release, memory_scope_device);
	      }
		  break;
		}
		if (local_flag == DEVICE_TO_TASK) {
		  int local_task_size = *(s_ctx.task_size);
		  kernel_ctx->num_groups = local_task_size;
		  atomic_store_explicit(&(kernel_ctx->completed), 0, memory_order_relaxed, memory_scope_device);
		  int lpg = *(s_ctx.participating_groups);
		  
		  // Wait until we have all the groups we need. This will be the response time.
		  // Because this first test is synchronous, we don't have to wait.
		  atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_GOT_GROUPS, memory_order_release, memory_scope_all_svm_devices);
          while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_relaxed, memory_scope_all_svm_devices) != DEVICE_TO_EXECUTE);
		  
		  // Send the tasks to do the task. 
		  for(int i = 0; i < local_task_size; i++) {
			  
			// the index is lpg - 1 because we're already missing one group because of the scheduler. 
			atomic_store_explicit(&(s_ctx.task_array[lpg - i]), TASK_MULT, memory_order_release, memory_scope_device);
	      }
		  
		  while (atomic_load_explicit(&(kernel_ctx->completed), memory_order_relaxed, memory_scope_device) != local_task_size);		  
		  atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING, memory_order_release, memory_scope_all_svm_devices);
		}
	  }
	}
  }
  
  // All other workgroups
  while(true) {
    int task = get_task(s_ctx, group_id);
    if (task == TASK_QUIT) {
	    break;
	}
	if (task == TASK_MULT) {
	  int kernel_group_id = *(s_ctx.participating_groups) - group_id;
	  MY_reduce(kernel_buffer, kernel_length, kernel_result, kernel_group_id, kernel_ctx->num_groups);
	  if (get_local_id(0) == 0) {
		  atomic_fetch_add(&(kernel_ctx->completed), 1);
		  atomic_store_explicit(&(s_ctx.task_array[group_id]), TASK_WAIT, memory_order_release, memory_scope_device);
	  }
	}
  }

}
//