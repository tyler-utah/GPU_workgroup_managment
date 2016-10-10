#pragma once

#include "discovery.h"

// kernel_ctx
typedef struct {
  ATOMIC_CL_INT_TYPE num_groups;
  CL_INT_TYPE group_ids[MAX_P_GROUPS];
  ATOMIC_CL_INT_TYPE executing_groups;
  MY_CL_GLOBAL Discovery_ctx *d_ctx;
} Kernel_ctx;