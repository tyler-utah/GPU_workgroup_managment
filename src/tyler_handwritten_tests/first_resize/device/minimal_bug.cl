typedef struct {
  CL_UCHAR_TYPE target;
  int i;
} Restoration_ctx;

typedef struct {

  atomic_int counter;
  atomic_int now_serving;

} Ticket_lock;

typedef struct {

  int count;
  int poll_open;
  int p_group_ids[512];
  Ticket_lock m;

} Discovery_ctx;

void TL_lock(Ticket_lock *m) {

  int ticket = atomic_fetch_add_explicit(&(m->counter), 1, memory_order_acq_rel,
                                         memory_scope_device);

  while (atomic_load_explicit(&(m->now_serving), memory_order_relaxed,
                              memory_scope_device) != ticket)
    ;

  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire,
                         memory_scope_device);
}

void TL_unlock(Ticket_lock *m) {

  int tmp = atomic_load_explicit(&(m->now_serving), memory_order_relaxed,
                                 memory_scope_device);

  tmp += 1;

  atomic_store_explicit(&(m->now_serving), tmp, memory_order_release,
                        memory_scope_device);
}

int p_get_group_id(__global Discovery_ctx *d_ctx) {
  return d_ctx->p_group_ids[get_group_id(0)];
}

int discovery_protocol_rep(__global Discovery_ctx *d_ctx) {

  int id;
  TL_lock(&(d_ctx->m));
  if ((d_ctx->poll_open)) {
    id = (d_ctx->count);
    (d_ctx->count) = id + 1;
    TL_unlock(&(d_ctx->m));
    d_ctx->p_group_ids[get_group_id(0)] = id;
  } else {
    TL_unlock(&(d_ctx->m));
    return 0;
  }

  TL_lock(&(d_ctx->m));

  if ((d_ctx->poll_open)) {
    d_ctx->poll_open = 0;
  }
  TL_unlock(&(d_ctx->m));

  return 1;
}

int discovery_protocol(__global Discovery_ctx *d_ctx, __local int *ret_flag) {
  int id = get_local_id(0);
  if (id == 0) {
    *ret_flag = discovery_protocol_rep(d_ctx);
  }
  barrier((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE));
  return *ret_flag;
}

typedef struct {
  atomic_int num_groups;
  int group_ids[512];
  atomic_int executing_groups;
  MY_CL_GLOBAL Discovery_ctx *d_ctx;
} Kernel_ctx;

int k_get_num_groups(__global Kernel_ctx *k_ctx) {
  return atomic_load_explicit(&(k_ctx->num_groups), memory_order_relaxed,
                              memory_scope_device);
}

int k_get_num_groups_acquire(__global Kernel_ctx *k_ctx) {
  return atomic_load_explicit(&(k_ctx->num_groups), memory_order_acquire,
                              memory_scope_device);
}

int k_get_global_size(__global Kernel_ctx *k_ctx) {
  return k_get_num_groups(k_ctx) * get_local_size(0);
}

int k_get_group_id(__global Kernel_ctx *k_ctx) {
  return k_ctx->group_ids[p_get_group_id(k_ctx->d_ctx)];
}

int k_get_global_id(__global Kernel_ctx *k_ctx) {
  return k_get_group_id(k_ctx) * get_local_size(0) + get_local_id(0);
}

typedef struct {

  MY_CL_GLOBAL int *participating_groups;

  MY_CL_GLOBAL atomic_int *task_array;

  MY_CL_GLOBAL atomic_int *scheduler_flag;

  MY_CL_GLOBAL int *task_size;

  MY_CL_GLOBAL atomic_int *available_workgroups;

  MY_CL_GLOBAL atomic_int *pool_lock;

  MY_CL_GLOBAL atomic_int *groups_to_kill;

  MY_CL_GLOBAL atomic_int *persistent_flag;

  MY_CL_GLOBAL Restoration_ctx *r_ctx_arr;

} CL_Scheduler_ctx;

int try_lock(atomic_int *m) {
  return !(atomic_exchange_explicit(m, 1, memory_order_relaxed,
                                    memory_scope_device));
}

void scheduler_lock(atomic_int *m) {
  while (atomic_exchange_explicit(m, 1, memory_order_relaxed,
                                  memory_scope_device) != 0)
    ;
  atomic_work_item_fence((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE),
                         memory_order_acquire, memory_scope_device);
}

void scheduler_unlock(atomic_int *m) {
  atomic_store_explicit(m, 0, memory_order_release, memory_scope_device);
}

int scheduler_needs_workgroups(CL_Scheduler_ctx s_ctx) {
  if (atomic_load_explicit(s_ctx.groups_to_kill, memory_order_relaxed,
                           memory_scope_device) > 0)
    return 1;
  return 0;
}

void scheduler_assign_tasks_graphics(CL_Scheduler_ctx s_ctx,
                                     __global Kernel_ctx *graphics_kernel_ctx) {

  int local_task_size = *(s_ctx.task_size);
  int lpg = *(s_ctx.participating_groups);

  for (int i = 0; i < local_task_size; i++) {

    graphics_kernel_ctx->group_ids[lpg - i] = i;

    while (atomic_load_explicit(&(s_ctx.task_array[lpg - i]),
                                memory_order_relaxed,
                                memory_scope_device) != -1)
      ;

    atomic_store_explicit(&(s_ctx.task_array[lpg - i]), 1, memory_order_release,
                          memory_scope_device);

    atomic_fetch_sub(s_ctx.available_workgroups, 1);
  }
}

void scheduler_assign_tasks_persistent(
    CL_Scheduler_ctx s_ctx, __global Kernel_ctx *persistent_kernel_ctx) {

  int local_task_size = *(s_ctx.task_size);
  int lpg = *(s_ctx.participating_groups);

  for (int i = 0; i < local_task_size; i++) {

    persistent_kernel_ctx->group_ids[i + 1] = i;

    while (atomic_load_explicit(&(s_ctx.task_array[i + 1]),
                                memory_order_relaxed,
                                memory_scope_device) != -1)
      ;

    s_ctx.r_ctx_arr[i + 1].target = 0;

    atomic_store_explicit(&(s_ctx.task_array[i + 1]), 2, memory_order_release,
                          memory_scope_device);

    atomic_fetch_sub(s_ctx.available_workgroups, 1);
  }
}

int get_task(CL_Scheduler_ctx s_ctx, int group_id, __local int *scratchpad,
             Restoration_ctx *r_ctx) {
  if (get_local_id(0) == 0) {

    int tmp;
    while (true) {
      tmp = atomic_load_explicit(&(s_ctx.task_array[group_id]),
                                 memory_order_relaxed, memory_scope_device);
      if (tmp != -1) {
        break;
      }
    }
    atomic_work_item_fence((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE),
                           memory_order_acquire, memory_scope_work_group);
    *scratchpad = tmp;
    *r_ctx = s_ctx.r_ctx_arr[group_id];
  }
  barrier((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE));
  return *scratchpad;
}

void scheduler_init(CL_Scheduler_ctx s_ctx, __global Discovery_ctx *d_ctx,
                    __global Kernel_ctx *graphics_kernel_ctx,
                    __global Kernel_ctx *persistent_kernel_ctx) {
  *(s_ctx.participating_groups) = d_ctx->count - 1;
  graphics_kernel_ctx->d_ctx = d_ctx;
  persistent_kernel_ctx->d_ctx = d_ctx;
  atomic_store_explicit(s_ctx.scheduler_flag, 1, memory_order_release,
                        memory_scope_all_svm_devices);
}

void scheduler_loop(CL_Scheduler_ctx s_ctx, __global Discovery_ctx *d_ctx,
                    __global Kernel_ctx *graphics_kernel_ctx,
                    __global Kernel_ctx *persistent_kernel_ctx) {

  while (true) {
    int local_flag =
        atomic_load_explicit(s_ctx.scheduler_flag, memory_order_relaxed,
                             memory_scope_all_svm_devices);

    if (local_flag == 2) {

      atomic_work_item_fence((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE),
                             memory_order_acquire,
                             memory_scope_all_svm_devices);

      for (int i = 0; i < 512; i++) {

        while (atomic_load_explicit(&(s_ctx.task_array[i]),
                                    memory_order_relaxed,
                                    memory_scope_device) != -1)
          ;
        atomic_store_explicit(&(s_ctx.task_array[i]), 0, memory_order_release,
                              memory_scope_device);
        atomic_fetch_sub(s_ctx.available_workgroups, 1);
      }
      break;
    }

    if (local_flag == 3) {
      atomic_work_item_fence((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE),
                             memory_order_acquire,
                             memory_scope_all_svm_devices);

      int local_task_size = *(s_ctx.task_size);
      atomic_store_explicit(&(graphics_kernel_ctx->num_groups), local_task_size,
                            memory_order_relaxed, memory_scope_device);
      atomic_store_explicit(&(graphics_kernel_ctx->executing_groups),
                            local_task_size, memory_order_relaxed,
                            memory_scope_device);

      scheduler_lock(s_ctx.pool_lock);

      int to_kill =
          local_task_size - atomic_load_explicit(s_ctx.available_workgroups,
                                                 memory_order_relaxed,
                                                 memory_scope_device);
      atomic_store_explicit(s_ctx.groups_to_kill, to_kill, memory_order_relaxed,
                            memory_scope_device);

      while (atomic_load_explicit(s_ctx.available_workgroups,
                                  memory_order_relaxed,
                                  memory_scope_device) < local_task_size)
        ;

      atomic_store_explicit(s_ctx.scheduler_flag, 4, memory_order_relaxed,
                            memory_scope_all_svm_devices);

      while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_relaxed,
                                  memory_scope_all_svm_devices) != 5)
        ;

      scheduler_assign_tasks_graphics(s_ctx, graphics_kernel_ctx);

      scheduler_unlock(s_ctx.pool_lock);

      while (atomic_load_explicit(&(graphics_kernel_ctx->executing_groups),
                                  memory_order_relaxed,
                                  memory_scope_device) != 0)
        ;

      atomic_store_explicit(s_ctx.scheduler_flag, 1, memory_order_release,
                            memory_scope_all_svm_devices);
    }

    if (local_flag == 6) {

      atomic_work_item_fence((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE),
                             memory_order_acquire,
                             memory_scope_all_svm_devices);

      int local_task_size = *(s_ctx.task_size);

      atomic_store_explicit(&(persistent_kernel_ctx->num_groups),
                            local_task_size, memory_order_relaxed,
                            memory_scope_device);

      atomic_store_explicit(&(persistent_kernel_ctx->executing_groups),
                            local_task_size, memory_order_relaxed,
                            memory_scope_device);

      int lpg = *(s_ctx.participating_groups);

      scheduler_lock(s_ctx.pool_lock);

      while (atomic_load_explicit(s_ctx.available_workgroups,
                                  memory_order_relaxed,
                                  memory_scope_device) < local_task_size)
        ;

      atomic_store_explicit(s_ctx.scheduler_flag, 4, memory_order_relaxed,
                            memory_scope_all_svm_devices);

      while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_relaxed,
                                  memory_scope_all_svm_devices) != 5)
        ;

      scheduler_assign_tasks_persistent(s_ctx, persistent_kernel_ctx);

      scheduler_unlock(s_ctx.pool_lock);

      atomic_store_explicit(s_ctx.scheduler_flag, 1, memory_order_release,
                            memory_scope_all_svm_devices);
    }
  }
}

typedef struct {

  atomic_int barrier_flags[512];

  atomic_int phase;

} IW_barrier;

void MY_reduce(int length, __global int *buffer, __global atomic_int *result,
               __global Kernel_ctx *kernel_ctx) {

}

void simple_barrier(

    __global IW_barrier *bar, __global Kernel_ctx *kernel_ctx,
    CL_Scheduler_ctx s_ctx, __local int *scratchpad, Restoration_ctx *r_ctx) {

}

/*---------------------------------------------------------------------------*/

__kernel void mega_kernel(

    int graphics_length, __global int *graphics_buffer,
    __global atomic_int *graphics_result,

    __global IW_barrier *bar,

    __global Discovery_ctx *d_ctx,

    __global Kernel_ctx *non_persistent_kernel_ctx,

    __global Kernel_ctx *persistent_kernel_ctx,

    __global atomic_int *scheduler_flag, __global int *scheduler_groups,
    __global atomic_int *task_array, __global int *task_size,
    __global atomic_int *available_workgroups, __global atomic_int *pool_lock,
    __global atomic_int *groups_to_kill, __global atomic_int *persistent_flag,
    __global Restoration_ctx *r_ctx_arr) {

  __local int scratchpad[2];
  if (discovery_protocol(d_ctx, scratchpad) == 0)
    return;
  ;

  CL_Scheduler_ctx s_ctx;
  s_ctx.scheduler_flag = scheduler_flag;
  s_ctx.participating_groups = scheduler_groups;
  s_ctx.task_array = task_array;
  s_ctx.task_size = task_size;
  s_ctx.available_workgroups = available_workgroups;
  s_ctx.pool_lock = pool_lock;
  s_ctx.groups_to_kill = groups_to_kill;
  s_ctx.persistent_flag = persistent_flag;
  s_ctx.r_ctx_arr = r_ctx_arr;

  int group_id = p_get_group_id(d_ctx);

  if (group_id == 0) {
    if (get_local_id(0) == 0) {

      scheduler_init(s_ctx, d_ctx, non_persistent_kernel_ctx,
                     persistent_kernel_ctx);

      scheduler_loop(s_ctx, d_ctx, non_persistent_kernel_ctx,
                     persistent_kernel_ctx);
    }
    barrier((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE));
    return;
  }

  Restoration_ctx r_ctx_local;

  while (true) {

    /* FIX: add a local barrier here */
    
    if (get_local_id(0) == 0) {
      atomic_store_explicit(&(s_ctx.task_array[group_id]), -1,
                            memory_order_relaxed, memory_scope_device);
      atomic_fetch_add(s_ctx.available_workgroups, 1);
    }

    int task = get_task(s_ctx, group_id, scratchpad, &r_ctx_local);

    if (task == 0) {
      break;
    }

    else if (task == 1) {

      // Would call non-persistent kernel - removed

      barrier((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE));

      if (get_local_id(0) == 0) {
        atomic_fetch_sub(&(non_persistent_kernel_ctx->executing_groups), 1);
      }
    }

    else if (task == 2) {

      // Would call persistent kernel - removed

      barrier((CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE));

      if (get_local_id(0) == 0) {
        int check =
            atomic_fetch_sub(&(persistent_kernel_ctx->executing_groups), 1);
        if (check == 1) {
          atomic_store_explicit(s_ctx.persistent_flag, 1, memory_order_relaxed,
                                memory_scope_all_svm_devices);
        }
      }
    }
  }
}
