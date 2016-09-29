#pragma once

#include "constants.h"

// For kernel code, define INT_TYPE as int, for host code cl_int
#ifndef INT_TYPE
#error "INT_TYPE not defined"
#endif

// For kernel code, define ATOMIC_INT_TYPE as atomic_int, for host code cl_int
#ifndef ATOMIC_INT_TYPE
#error "ATOMIC_INT_TYPE not defined"
#endif

// Should be defined somewhere else.
#ifndef MAX_P_GROUPS
#error "MAX_P_GROUPS not defined"
#endif

#include "ticket_lock.h"

// discovery protocol context
typedef struct {

  INT_TYPE count;
  INT_TYPE poll_open;
  INT_TYPE p_group_ids[MAX_P_GROUPS];
  Ticket_lock m;

} Discovery_ctx;