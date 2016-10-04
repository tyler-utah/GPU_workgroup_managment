#pragma once

#include "../rt_common/iw_barrier.h"
#include "kernel_ctx.cl"

void global_barrier(__global IW_barrier *bar, __global Kernel_ctx *kernel_ctx) {
  
  int id = k_get_group_id(kernel_ctx);
	
  // Master workgroup
  if (id == 0) {
    for (int peer_block = get_local_id(0) + 1;
         peer_block < k_get_num_groups(kernel_ctx);
         peer_block += get_local_size(0)) {
			 
	  // Wait for the slave
      while (atomic_load_explicit(&(bar->barrier_flags[peer_block]), memory_order_relaxed, memory_scope_device) == 0);
	  
	   // Synchronise
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
    }
	
	BARRIER;
	
	for (int peer_block = get_local_id(0) + 1;
         peer_block < k_get_num_groups(kernel_ctx);
         peer_block += get_local_size(0)) {

	  // Release slaves
      atomic_store_explicit(&(bar->barrier_flags[peer_block]), 0, memory_order_release, memory_scope_device);
    }
  }
	
  // Slave workgroups
  else {
	BARRIER;
	
	if (get_local_id(0) == 0) {
	  // Mark arrival
      atomic_store_explicit(&(bar->barrier_flags[id]), 1, memory_order_release, memory_scope_device);

      // Wait to be released by the master
      while (atomic_load_explicit(&(bar->barrier_flags[id]), memory_order_relaxed, memory_scope_device) == 1);

      // Synchronise
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
	}
	
	BARRIER;
  }
}
