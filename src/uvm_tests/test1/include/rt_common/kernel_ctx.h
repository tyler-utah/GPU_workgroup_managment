#pragma once

// kernel_ctx
typedef struct {
  CL_INT_TYPE num_groups;
  CL_INT_TYPE group_ids[MAX_P_GROUPS];
  ATOMIC_CL_INT_TYPE completed;
} Kernel_ctx;