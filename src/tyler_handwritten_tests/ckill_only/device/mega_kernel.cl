
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

// Simple min reduce from:
// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
void MY_reduce(int length,
            __global int* buffer,
            __global atomic_int* result,
			__global Kernel_ctx *kernel_ctx) {
				
  __local int scratch[256];
  int gid = k_get_global_id(kernel_ctx);
  int stride = k_get_global_size(kernel_ctx);
  int local_index = get_local_id(0);
  
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

void simple_barrier(__global IW_barrier * bar, __global Kernel_ctx *kernel_ctx, CL_Scheduler_ctx s_ctx, __local int* scratchpad) {
	
	
  // local variable that needs to be restored for cfork();
  int i = 0;
  
  // Just loop doing a global barrier for N iterations and then quit
  while (true) {
    i++;
	
    int wg_num = global_barrier_ckill(bar, kernel_ctx, s_ctx, scratchpad);
	if (wg_num == -1) {
		return;
	}
	
	if (i == 100000) {
		break;
	}
  
  }
}

__kernel void mega_kernel(
                          // Graphics kernel args
						  int graphics_length,
						  __global int * graphics_buffer,
						  __global atomic_int * graphics_result,
						  
						  // Persistent kernel args just need the barrier
						  __global IW_barrier *bar,
						  
						  // Discovery context
						  __global Discovery_ctx *d_ctx,
						  
						  // Kernel context for graphics kernel
                          __global Kernel_ctx *graphics_kernel_ctx,
						  
						  // Kernel context for persistent kernel
						  __global Kernel_ctx *persistent_kernel_ctx,
						  
						  // Scheduler args need to be passed individually
                          SCHEDULER_ARGS
						  ){
							 
							 
  __local int scratchpad;
  DISCOVERY_PROTOCOL(d_ctx);
  
  // Scheduler init (makes a variable named s_ctx)
  INIT_SCHEDULER;

  int group_id = p_get_group_id(d_ctx); 

  // Scheduler workgroup
  if (group_id == 0) {
    if (get_local_id(0) == 0) {
	
      // Do any initialisation here before the main loop.
	  scheduler_init(s_ctx, d_ctx, graphics_kernel_ctx, persistent_kernel_ctx);
	
	  // Loops forever waiting for signals from the host. Host can issue a quit signal though.
	  scheduler_loop(s_ctx, d_ctx, graphics_kernel_ctx, persistent_kernel_ctx);
	  
	}
	BARRIER;
	return;
  }
  
  
  // All other workgroups
  
  Restoration_ctx r_ctx_local;
  while(true) {
	  
	// Workgroups are initially available
    if (get_local_id(0) == 0) {
	  atomic_store_explicit(&(s_ctx.task_array[group_id]), TASK_WAIT, memory_order_relaxed, memory_scope_device);
      atomic_fetch_add(s_ctx.available_workgroups, 1);
    }
	
	// This is synchronous, returns QUIT, MULT, or PERSIST tasks
    int task = get_task(s_ctx, group_id, &scratchpad, &r_ctx_local);
	
	// Quit is easy
    if (task == TASK_QUIT) {
	  break;
	}
	
	// The traditional task.
	else if (task == TASK_MULT) {
	  
	  MY_reduce(graphics_length, graphics_buffer, graphics_result, graphics_kernel_ctx);
	  
	  // One representative group states that we're not currently executing
	  BARRIER;
	  
	  // One representative states that we've completed the kernel
	  if (get_local_id(0) == 0) {
	    atomic_fetch_sub(&(graphics_kernel_ctx->executing_groups), 1);
	  }
	}
    
	// The persistent task.
    else if (task == TASK_PERSIST) {
	  
	  // We can exit either 0 (normal exit) or ckilled() exit of -1
	  simple_barrier(bar, persistent_kernel_ctx, s_ctx, &scratchpad);
	  
	  // Wait for all threads in the workgroup to reach this point
	  BARRIER;
	  
	  // One representative group states that we're not currently executing
	  if (get_local_id(0) == 0) {
	    int check = atomic_fetch_sub(&(persistent_kernel_ctx->executing_groups), 1);
		if (check == 1) {
          atomic_store_explicit(s_ctx.persistent_flag, PERSIST_TASK_DONE, memory_order_relaxed, memory_scope_all_svm_devices);
		}
	  }
	}
  }
}
//