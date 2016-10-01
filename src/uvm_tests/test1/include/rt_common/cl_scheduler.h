#pragma once

// For kernel code, define INT_TYPE as int, for host code cl_int
#ifndef CL_INT_TYPE
#error "CL_INT_TYPE not defined"
#endif

// For kernel code, define ATOMIC_INT_TYPE as atomic_int, for host code cl_int
#ifndef ATOMIC_CL_INT_TYPE
#error "ATOMIC_CL_INT_TYPE not defined"
#endif

#define DEVICE_SCHEDULER_INIT 0
#define DEVICE_WAITING 1
#define DEVICE_TO_QUIT 2
#define DEVICE_TO_TASK 3
#define DEVICE_GOT_GROUPS 4
#define DEVICE_TO_EXECUTE 5


// scheduler ctx
typedef struct {
  MY_CL_GLOBAL CL_INT_TYPE * participating_groups;
  MY_CL_GLOBAL ATOMIC_CL_INT_TYPE * scheduler_flag;
  MY_CL_GLOBAL ATOMIC_CL_INT_TYPE * task_array;
  MY_CL_GLOBAL CL_INT_TYPE * task_size;
} CL_Scheduler_ctx;