
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

__kernel void simple_barrier(__global IW_barrier * bar, __global Kernel_ctx *kernel_ctx) {
	
	
  // local variable that needs to be restored for cfork();
  int i = 0;
  
  // Just loop doing a global barrier for N iterations and then quit
  while (true) {
    i++;
	
    global_barrier(bar, kernel_ctx);
	
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
  

  while(true) {
	  
	// Workgroups are initially available
    if (get_local_id(0) == 0) {
      atomic_fetch_add(s_ctx.available_workgroups, 1);
    }
	
	// This is synchronous
    int task = get_task(s_ctx, group_id);
	
    if (task == TASK_QUIT) {
	    break;
	}
	if (task == TASK_MULT) {
	  
	  MY_reduce(graphics_length, graphics_buffer, graphics_result, graphics_kernel_ctx);
	  
	  BARRIER;
	  
	  if (get_local_id(0) == 0) {
		  atomic_fetch_add(&(graphics_kernel_ctx->completed), 1);
		  
		  atomic_store_explicit(&(s_ctx.task_array[group_id]), TASK_WAIT, memory_order_relaxed, memory_scope_device);
	  }
	}
  
    if (task == TASK_PERSIST) {
	  
	  simple_barrier(bar, persistent_kernel_ctx);  
	  BARRIER;
	  
	  if (get_local_id(0) == 0) {
		  atomic_fetch_add(&(persistent_kernel_ctx->completed), 1);
		  
		  atomic_store_explicit(&(s_ctx.task_array[group_id]), TASK_WAIT, memory_order_relaxed, memory_scope_device);
	  }
	}
  }

}
//