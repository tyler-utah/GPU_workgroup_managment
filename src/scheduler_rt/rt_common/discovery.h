#pragma once

#include "constants.h"

// For kernel code, define INT_TYPE as int, for host code cl_int
#ifndef CL_INT_TYPE
#error "CL_INT_TYPE not defined"
#endif

// For kernel code, define ATOMIC_INT_TYPE as atomic_int, for host code cl_int
#ifndef ATOMIC_CL_INT_TYPE
#error "ATOMIC_CL_INT_TYPE not defined"
#endif

// Should be defined somewhere else.
#ifndef MAX_P_GROUPS
#error "MAX_P_GROUPS not defined"
#endif

#include "ticket_lock.h"

// discovery protocol context
typedef struct {

  CL_INT_TYPE count;
  CL_INT_TYPE poll_open;
  CL_INT_TYPE p_group_ids[MAX_P_GROUPS];
  Ticket_lock m;
  
  // Flag to immediately exit, when we're just getting the occupancy
  CL_INT_TYPE im_exit;

} Discovery_ctx;