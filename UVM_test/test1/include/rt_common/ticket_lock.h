#pragma once

// For kernel code, define INT_TYPE as int, for host code cl_int
#ifndef INT_TYPE
#error "INT_TYPE not defined"
#endif

// For kernel code, define ATOMIC_INT_TYPE as atomic_int, for host code cl_int
#ifndef ATOMIC_INT_TYPE
#error "ATOMIC_INT_TYPE not defined"
#endif

// discovery protocol context
typedef struct {

  ATOMIC_INT_TYPE counter;
  ATOMIC_INT_TYPE now_serving;

} Ticket_lock;