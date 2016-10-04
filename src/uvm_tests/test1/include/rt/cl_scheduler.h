#pragma once

#include "../rt_common/cl_scheduler.h"
#include "cl_execution.h"


void mk_init_scheduler_ctx(CL_Execution *exec, CL_Scheduler_ctx *s_ctx) {
  s_ctx->participating_groups = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(cl_int), 4);
  s_ctx->scheduler_flag = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
  s_ctx->task_size = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
  s_ctx->task_array = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_READ_WRITE, sizeof(cl_int) * MAX_P_GROUPS, 4);
  s_ctx->available_workgroups = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_READ_WRITE, sizeof(cl_int), 4);


  for (int i = 0; i < MAX_P_GROUPS; i++) {
	  s_ctx->task_array[i] = TASK_WAIT;
  }
  
  *(s_ctx->scheduler_flag) = DEVICE_SCHEDULER_INIT;
  *(s_ctx->available_workgroups) = 0;
}

void free_scheduler_ctx(CL_Execution *exec, CL_Scheduler_ctx *s_ctx) {
  clSVMFree(exec->exec_context(), s_ctx->participating_groups);
  clSVMFree(exec->exec_context(), s_ctx->scheduler_flag);
  clSVMFree(exec->exec_context(), s_ctx->task_array);
  clSVMFree(exec->exec_context(), s_ctx->task_size);
  clSVMFree(exec->exec_context(), s_ctx->available_workgroups);
}

int set_scheduler_args(cl::Kernel *k, CL_Scheduler_ctx *s_ctx, int &arg_index) {
  int err = 0;
  err =  clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->scheduler_flag);
  arg_index++;
  err |= clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->participating_groups);
  arg_index++;
  err |= clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->task_array);
  arg_index++;
  err |= clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->task_size);
  arg_index++;
  err |= clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->available_workgroups);
  arg_index++;
  return err;
}