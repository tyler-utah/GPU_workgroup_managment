#pragma once

// kernel_ctx
typedef struct {

  // for the XF barrier
  ATOMIC_CL_INT_TYPE barrier_flags[MAX_P_GROUPS];

  // Extra member in case we want to play with a phase reversal barrier
  // (likely for the arbitrary scheduler)
  ATOMIC_CL_INT_TYPE phase;
  CL_INT_TYPE num_groups;
  CL_INT_TYPE to_kill;

  /* Hugues: for sense reversal barrier (I do not use the variables
     above since I need atomic_int for counter, and num_groups is
     currently used by the barriers implem) */
  ATOMIC_CL_INT_TYPE counter;
  ATOMIC_CL_INT_TYPE sense;

} IW_barrier;
