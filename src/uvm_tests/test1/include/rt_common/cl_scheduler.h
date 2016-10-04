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
#define DEVICE_TO_PERSISTENT_TASK 6

// scheduler ctx
typedef struct {
	
  // Number of participating groups
  MY_CL_GLOBAL CL_INT_TYPE * participating_groups;
  
  // Tasks for participating groups to take
  MY_CL_GLOBAL ATOMIC_CL_INT_TYPE * task_array;
  
  // Flag to communicate with the host with
  MY_CL_GLOBAL ATOMIC_CL_INT_TYPE * scheduler_flag;
  
  // How many workgroups to execute the task with
  MY_CL_GLOBAL CL_INT_TYPE * task_size;
  
  // How many workgroups are currently
  MY_CL_GLOBAL ATOMIC_CL_INT_TYPE * available_workgroups;
  
} CL_Scheduler_ctx;