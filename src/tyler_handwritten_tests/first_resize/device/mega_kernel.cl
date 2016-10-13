
// It is important to include this first because other files use it.
#include "restoration_ctx.h"

#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

// This is the "graphics kernel"
void MY_reduce(int length,
            __global int* buffer,
            __global atomic_int* result,
			
			// New arg, the kernel ctx
			__global Kernel_ctx *kernel_ctx) {
				
  __local int scratch[256];
  int gid = k_get_global_id(kernel_ctx);
  int stride = k_get_global_size(kernel_ctx);
  int local_index = get_local_id(0);
  
  for (int global_index = gid; global_index < length; global_index += stride) {
    
    if (global_index < length) {
      scratch[local_index] = buffer[global_index];
    } else {
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
	
    if (local_index == 0) {
	  atomic_fetch_min((result), scratch[0]);
    }
  }
}

// Persistent Kernel
void simple_barrier(
                    // All these are new arguments. 
		            __global IW_barrier * bar, 
		            __global Kernel_ctx *kernel_ctx, 
		            CL_Scheduler_ctx s_ctx, 
		            __local int* scratchpad, 
		            Restoration_ctx *r_ctx) {
		
  // local variable that needs to be restored for cfork();
  int i = 0;
  
  
  int scheduler_done = 0;
  
  if (r_ctx->target != 0) {
    i = r_ctx->i;
  }
  
  // Just loop doing a global barrier for N iterations and then quit
  while (!scheduler_done) {
    switch (r_ctx->target) {
	case 0:
	  if (!(true)) {
	    scheduler_done = true;
		break;
	  }
	  
      i++;
	  
	  
	  // Make a restoration context for this barrier. 
	  Restoration_ctx to_fork;
	  to_fork.target = 1;
	  to_fork.i = i;
	  
	  global_barrier_resize(bar, kernel_ctx, s_ctx, scratchpad, &to_fork);
	  
	case 1:
	    r_ctx->target = 0;
		

	  if (i == 100000) {
	    scheduler_done = true;
		break;
	  }
    }
  }
}

__kernel void mega_kernel(
                          // Graphics kernel args
						  int graphics_length,
						  __global int * graphics_buffer,
						  __global atomic_int * graphics_result,
						  
						  // Persistent kernel args (empty for this example)
						  
						  // Barrier object
						  __global IW_barrier *bar,
						  
						  // Discovery context
						  __global Discovery_ctx *d_ctx,
						  
						  // Kernel context for graphics kernel
                          __global Kernel_ctx *non_persistent_kernel_ctx,
						  
						  // Kernel context for persistent kernel
						  __global Kernel_ctx *persistent_kernel_ctx,
						  
						  // Scheduler args need to be passed individually
                          SCHEDULER_ARGS
						  ){

// These need to be made by the kernel merge tool. Its the original graphics kernel with the non_persistent_kernel_ctx as a final arg.						  
#define NON_PERSISTENT_KERNEL MY_reduce(graphics_length, graphics_buffer, graphics_result, non_persistent_kernel_ctx)

// This is the original persistent kernel with the bar, persistent_kernel_ctx, s_ctx, scratchpad, and (by pointer) local restoration context.
#define PERSISTENT_KERNEL simple_barrier(bar, persistent_kernel_ctx, s_ctx, scratchpad, &r_ctx_local);

// Everything else is in here	
#include "main_device_body.cl"
}
//