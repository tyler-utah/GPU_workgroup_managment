#pragma once

// For kernel code, define INT_TYPE as int, for host code cl_int
#ifndef CL_INT_TYPE
#error "CL_INT_TYPE not defined"
#endif

// For kernel code, define ATOMIC_INT_TYPE as atomic_int, for host code cl_int
#ifndef ATOMIC_CL_INT_TYPE
#error "ATOMIC_CL_INT_TYPE not defined"
#endif

// discovery protocol context
typedef struct {

  ATOMIC_CL_INT_TYPE counter;
  ATOMIC_CL_INT_TYPE now_serving;

} Ticket_lock;