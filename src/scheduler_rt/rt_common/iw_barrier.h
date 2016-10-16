#pragma once

// kernel_ctx
typedef struct {
	
  // for the XF barrier
  ATOMIC_CL_INT_TYPE barrier_flags[MAX_P_GROUPS];
  
  // Extra member in case we want to play with a phase reversal barrier
  // (likely for the arbitrary scheduler)
  ATOMIC_CL_INT_TYPE phase;
  CL_INT_TYPE num_groups;
  
} IW_barrier;