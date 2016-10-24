
// It is important to include this first because other files use it.
#include "../rt_common/cl_types.h"
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

// mega_kernel: combines the color1 and color2 kernels using an
// inter-workgroup barrier and the discovery protocol
void color_persistent    ( __global int   *row,                        //0
                           __global int   *col,                        //1
                           __global float *node_value,                 //2
                           __global int   *color_array,                //3
                           __global int   *stop1,                      //4
                           __global int   *stop2,                      //5
                           __global float *max_d,                      //6
                           const  int num_nodes,                       //7
                           const  int num_edges,                       //8
						   
						  __global IW_barrier * bar, 
		                  __global Kernel_ctx *kernel_ctx, 
		                  CL_Scheduler_ctx s_ctx, 
		                  __local int* scratchpad, 
		                  Restoration_ctx *r_ctx
                           ) {                                         


  __global int * write_stop = stop1;
  __global int * read_stop = stop2;
  __global int * swap;
  int graph_color = 1;
  int i = 0;
  int tid;
  int stride;

  	    //if (k_get_group_id(kernel_ctx) == 7) {
	     // atomic_store(s_ctx.check_value, );
        //}
  
  if (r_ctx->target != 0) {
    write_stop = r_ctx->write_stop;
	read_stop = r_ctx->read_stop;
	swap = r_ctx->swap;
    graph_color = r_ctx->graph_color;
	i = r_ctx->i;
  }

  Restoration_ctx to_fork;
  
  while (graph_color < 200) {
	switch(r_ctx->target) {
    case 0:
	//if (r_ctx->target < 1) {

	  if (!true) {
		return;
	  }

      
	  // Get global participating group id and the stride
      tid = b_get_global_id(bar, kernel_ctx);
      stride = b_get_global_size(bar, kernel_ctx);
	  //tid = k_get_global_id(kernel_ctx);
	  //stride = k_get_global_size(kernel_ctx);
	  
      // Original application --- color --- start

      // The original kernels used an 'if' here. We need a 'for' loop
      for (int i = tid; i < num_nodes; i+=stride) {


        // If the vertex is still not colored
        if (color_array[i] == -1) {

          // Get the start and end pointer of the neighbor list
          int start = row[i];
          int end;
          if (i + 1 < num_nodes)
            end = row[i + 1];
          else
            end = num_edges;

          float maximum = -1;

          // Navigate the neighbor list
          for (int edge = start; edge < end; edge++) {

            // Determine if the vertex value is the maximum in the neighborhood
            if (color_array[col[edge]] == -1 && start != end - 1) {
              *write_stop = 1;
              if (node_value[col[edge]] > maximum)
                maximum = node_value[col[edge]];
            }
          }
          // Assign maximum the max array
          max_d[i] = maximum;
        }
      }
	  

      // Two terminating variables allow us to only use 1
      // inter-workgroup barrier and still avoid a data-race
      swap = read_stop;
      read_stop = write_stop;
      write_stop = swap;

      // Original application --- color --- end
	  
	  to_fork.write_stop = write_stop;
	  to_fork.read_stop = read_stop;
	  to_fork.swap = swap;
	  to_fork.graph_color = graph_color;
  	  to_fork.target = 1;
	  to_fork.i = i;


      // Inter-workgroup barrier
      global_barrier_resize(bar, kernel_ctx, s_ctx, scratchpad, &to_fork);
	//}
	case 1:
    //if (r_ctx->target < 2) {
	    r_ctx->target = 0;
		
		
	  tid = b_get_global_id(bar, kernel_ctx);
      stride = b_get_global_size(bar, kernel_ctx);
	  
	  //tid = k_get_global_id(kernel_ctx);
	  //stride = k_get_global_size(kernel_ctx);
	  
		
      // Original application --- color2 --- start

      // The original kernels used an 'if' here. We need a 'for' loop
      for (int i = tid; i < num_nodes; i+=stride) {

        // If the vertex is still not colored
        if (color_array[i] == -1) {
          if (node_value[i] > max_d[i])

            // Assign a color
            color_array[i] = graph_color;
        }
      }

	  
      if (*read_stop == 0) {
		 atomic_store_explicit(s_ctx.check_value, 0, memory_order_relaxed, memory_scope_all_svm_devices);
       //*(s_ctx.participating_groups) = -1;
	   //break;
	   return;
      }

      graph_color = graph_color + 1;
      *write_stop = 0;

      // Original application --- color2 --- end
	  
	  to_fork.write_stop = write_stop;
	  to_fork.read_stop = read_stop;
	  to_fork.swap = swap;
	  to_fork.graph_color = graph_color;
	  to_fork.target = 2;
	  to_fork.i = i;

      // Inter-workgroup barrier
      global_barrier_resize(bar, kernel_ctx, s_ctx, scratchpad, &to_fork);
	//}
	//if (r_ctx->target < 3) {
	  case 2:
	  r_ctx->target = 0;
	  

	  		
		
	  i++;
    //}
	}

  }
}

__kernel void mega_kernel(
                          // Graphics kernel args
						  int graphics_length,
						  __global int * graphics_buffer,
						  __global atomic_int * graphics_result,
						  
						  // Persistent kernel args (empty for this example)
						   __global int   *row,                        //0
                           __global int   *col,                        //1
                           __global float *node_value,                 //2
                           __global int   *color_array,                //3
                           __global int   *stop1,                      //4
                           __global int   *stop2,                      //5
                           __global float *max_d,                      //6
                           const  int num_nodes,                       //7
                           const  int num_edges,                       //8
						   
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

// These need to be made by the kernel merge tool. Its the original graphics kernel with the graphics_kernel_ctx as a final arg.						  
#define NON_PERSISTENT_KERNEL MY_reduce(graphics_length, graphics_buffer, graphics_result, non_persistent_kernel_ctx)

// This is the original persistent kernel with the bar, persistent_kernel_ctx, s_ctx, scratchpad, and (by pointer) local restoration context.
#define PERSISTENT_KERNEL color_persistent(row, col, node_value, color_array, stop1, stop2, max_d, num_nodes, num_edges, bar, persistent_kernel_ctx, s_ctx, scratchpad, &r_ctx_local);

// Everything else is in here	
#include "main_device_body.cl"
}
//