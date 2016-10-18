#pragma once

#include "../rt_common/cl_scheduler.h"
#include "cl_execution.h"


void mk_init_scheduler_ctx(CL_Execution *exec, CL_Scheduler_ctx *s_ctx) {
  s_ctx->participating_groups = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(cl_int), 4);
  s_ctx->scheduler_flag = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
  s_ctx->task_size = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
  s_ctx->task_array = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int) * MAX_P_GROUPS, 4);
  s_ctx->available_workgroups = (cl_int*) clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
  s_ctx->pool_lock = (cl_int*)clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
  s_ctx->groups_to_kill = (cl_int*)clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
  s_ctx->persistent_flag = (cl_int*)clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(cl_int), 4);
  s_ctx->r_ctx_arr = (Restoration_ctx*)clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(Restoration_ctx) * MAX_P_GROUPS, 4);
  s_ctx->check_value = (cl_int*)clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(cl_int), 4);


  *(s_ctx->persistent_flag) = PERSIST_TASK_UNDEF;
  *(s_ctx->groups_to_kill) = 0;
  *(s_ctx->pool_lock) = 0;
  *(s_ctx->check_value) = -1;

  for (int i = 0; i < MAX_P_GROUPS; i++) {
	  s_ctx->task_array[i] = TASK_UNINIT;
	  s_ctx->r_ctx_arr[i].target = 0;
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
  clSVMFree(exec->exec_context(), s_ctx->pool_lock);
  clSVMFree(exec->exec_context(), s_ctx->groups_to_kill);
  clSVMFree(exec->exec_context(), s_ctx->persistent_flag);
  clSVMFree(exec->exec_context(), s_ctx->r_ctx_arr);
  clSVMFree(exec->exec_context(), s_ctx->check_value);
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
  err |= clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->pool_lock);
  arg_index++;
  err |= clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->groups_to_kill);
  arg_index++;
  err |= clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->persistent_flag);
  arg_index++;
  err |= clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->r_ctx_arr);
  arg_index++;
  err |= clSetKernelArgSVMPointer((*k)(), arg_index, s_ctx->check_value);
  arg_index++;
  return err;
}