 // To be included in the "main" body of the device. 
 // Requires two macros to be defined:
 // NON_PERSISTENT_KERNEL and PERSISTENT_KERNEL
  
  
  __local int scratchpad[2];
  __local Restoration_ctx lm_r_ctx;
  
  DISCOVERY_PROTOCOL(d_ctx, scratchpad);
  
  // Scheduler init (makes a variable named s_ctx)
  INIT_SCHEDULER;

  int group_id = p_get_group_id(d_ctx); 

  // Scheduler workgroup
  if (group_id == 0) {
    if (get_local_id(0) == 0) {
		
	
      // Do any initialisation here before the main loop.
	  scheduler_init(s_ctx, d_ctx, non_persistent_kernel_ctx, persistent_kernel_ctx);
	
	  // Loops forever waiting for signals from the host. Host can issue a quit signal though.
	  scheduler_loop(s_ctx, d_ctx, non_persistent_kernel_ctx, persistent_kernel_ctx, bar);
	  
	}
	BARRIER;
	return;
  }
  
  // All other workgroups
  
  Restoration_ctx r_ctx_local;
  
  while(true) {

  // Tyler: Do not remove! This doesn't seem to be semantically required,
	// but if its not included then intel 500/520 GPUs deadlock. Probably
	// due to some compiler re-ordering, etc.
	BARRIER;
	  
	// Workgroups are initially available
    if (get_local_id(0) == 0) {
	  atomic_store_explicit(&(s_ctx.task_array[group_id]), TASK_WAIT, memory_order_relaxed, memory_scope_device);
      atomic_fetch_add(s_ctx.available_workgroups, 1);
    }
	
	
	// This is synchronous, returns QUIT, MULT, or PERSIST tasks
    int task = get_task(s_ctx, group_id, scratchpad, &r_ctx_local, &lm_r_ctx);
	
	// Quit is easy
    if (task == TASK_QUIT) {
	  break;
	}
	
	// The traditional task.
	else if (task == TASK_MULT) {
	  
	  // Launch the non-persistent kernel
	  NON_PERSISTENT_KERNEL;
	  
	  // One representative group states that we're not currently executing
	  BARRIER;
	  
	  // One representative states that we've completed the kernel
	  if (get_local_id(0) == 0) {
	    atomic_fetch_sub(&(non_persistent_kernel_ctx->executing_groups), 1);
	  }
	}
    
	// The persistent task.
    else if (task == TASK_PERSIST) {
			  
	    
	  PERSISTENT_KERNEL;
	  
	  // Wait for all threads in the workgroup to reach this point
	  BARRIER;
	  
	  // One representative group states that we're not currently executing
	  if (get_local_id(0) == 0) {
	    //int check = atomic_fetch_sub_explicit(&(persistent_kernel_ctx->executing_groups), 1, memory_order_relaxed, memory_scope_device);
		atomic_fetch_sub_explicit(&(persistent_kernel_ctx->executing_groups), 1, memory_order_relaxed, memory_scope_device);
		//if (check == 1) {
        //  atomic_store_explicit(s_ctx.persistent_flag, PERSIST_TASK_DONE, memory_order_seq_cst, memory_scope_all_svm_devices);
		//}
		int check = atomic_fetch_sub_explicit(s_ctx.persistent_flag, 1, memory_order_release, memory_scope_all_svm_devices);

	  }
	}
  }