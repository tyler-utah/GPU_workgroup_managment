#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

/*
  Matrix multiplication: C = A * B

  We assume C is big enough to store the result.
*/

/*---------------------------------------------------------------------------*/

int hash_mat(int *M, int num_row, int num_col) {
  // hash the diagonal using djb2, see
  // http://www.cse.yorku.ca/~oz/hash.html
  int hash = 5381;
  int row = 0;
  int col = 0;
  while (row < num_row && col < num_col) {
    hash = (hash * 33) + M[(row * num_col) + col];
    row++;
    col++;
  }
  return hash;
}

/*---------------------------------------------------------------------------*/

void matmult(__global int *A, const int A_row, const int A_col, __global int *B,
             const int B_row, const int B_col, __global int *C,
             __global atomic_int *counter, __global atomic_int *hash,
             __global Kernel_ctx *__k_ctx) {
  /* safety */
  if (A_col != B_row) {
    return;
  }

  int gid = k_get_global_id(__k_ctx);
  int num_threads = k_get_global_size(__k_ctx);
  int c_size = A_row * B_col;

  /* Multiply matrices */
  for (int i = gid; i < c_size; i += num_threads) {
    int c_row = i / B_col;
    int c_col = i % B_col;
    int a_offset = c_row * A_col;
    C[i] = 0;
    for (int j = 0; j < B_row; j++) {
      C[i] += A[a_offset + j] * B[(j * B_col) + c_col];
    }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (get_local_id(0) == 0) {
    int finished = atomic_fetch_add(counter, 1);
    if (finished == (k_get_num_groups(__k_ctx) - 1)) {
      int h = hash_mat(C, A_row, B_col);
      atomic_store(hash, h);
    }
  }
}

/*---------------------------------------------------------------------------*/

__global int __junk_global;

/* Hugues: octree_common.h hard-coded-included since kernel_merge does
   not provide option to indicate additionnal include dirs/files */

/* TODO
   - use pool with mutexes, cf connect_four
   - use task donation when overflow of pool
   - test with big number of particles
 */

/*---------------------------------------------------------------------------*/

typedef struct {
  float4 middle;
  bool flip;
  uint end;
  uint beg;
  uint treepos;
} Task;

/*---------------------------------------------------------------------------*/

typedef struct {
  atomic_int tail;
  atomic_int head;
} DequeHeader;

/*---------------------------------------------------------------------------*/

int pool_try_lock(__global atomic_int *task_pool_lock, int pool_id) {
  int expected = false;
  return atomic_compare_exchange_strong(&(task_pool_lock[pool_id]), &expected,
                                        true);
}

/*---------------------------------------------------------------------------*/

void pool_unlock(__global atomic_int *task_pool_lock, int pool_id) {
  atomic_store(&(task_pool_lock[pool_id]), false);
}

/*---------------------------------------------------------------------------*/

/* wgm_task_pop() MUST be called by local_id 0. This function grabs a
   task from a pool and stores it in the task argument. Returns true if
   it works.  must local_fence after this function */
int wgm_task_pop(__local Task *task, __global Task *pools,
                 __global atomic_int *task_pool_lock, __global int *pool_head,
                 const int pool_size, int pool_id) {
  int poped = false;
  /* spinwait on the pool lock */
  while (!(pool_try_lock(task_pool_lock, pool_id)))
    ;
  /* If pool is not empty, pick up the latest inserted task. */
  if (pool_head[pool_id] > 0) {
    atomic_dec(&(pool_head[pool_id]));
    *task = pools[(pool_size * pool_id) + pool_head[pool_id]];
    poped = true;
  }
  pool_unlock(task_pool_lock, pool_id);
  return poped;
}

/*---------------------------------------------------------------------------*/

/* wgm_task_push() adds the task argument to the indicated pool. Returns
   true if it worked. */
int wgm_task_push(__local Task *task, __global Task *pools,
                  __global atomic_int *task_pool_lock, __global int *pool_head,
                  const int pool_size, int pool_id) {
  int pushed = false;
  /* spinwait on the pool lock */
  while (!(pool_try_lock(task_pool_lock, pool_id)))
    ;
  /* If pool is not full, insert task */
  if (pool_head[pool_id] < pool_size) {
    pools[(pool_size * pool_id) + pool_head[pool_id]] = *task;
    atomic_inc(&(pool_head[pool_id]));
    pushed = true;
  }
  pool_unlock(task_pool_lock, pool_id);
  return pushed;
}

/*---------------------------------------------------------------------------*/

__constant int mc[8][3] = {{-1, -1, -1}, {+1, -1, -1}, {-1, +1, -1},
                           {+1, +1, -1}, {-1, -1, +1}, {+1, -1, +1},
                           {-1, +1, +1}, {+1, +1, +1}};

/*---------------------------------------------------------------------------*/

int whichbox(volatile float4 pos, float4 middle) {
  int box = pos.x < middle.x ? 0 : 1;
  box += pos.y < middle.y ? 0 : 2;
  box += pos.z < middle.z ? 0 : 4;
  return box;
}
/*
  0   - - -
  1   + - -
  2   - + -
  3   + + -
  4   - - +
  5   + - +
  6   - + +
  7   + + +
*/

/*---------------------------------------------------------------------------*/

void octree_init(__global Task *pools, __global atomic_int *task_pool_lock,
                 __global int *pool_head, const int num_pools,
                 const int pool_size, __global atomic_uint *treeSize,
                 __global atomic_uint *particlesDone, unsigned int numParticles,
                 __local Task *t) {
  /* reset pools, no need to lock here since only global master thread
     runs this function */
  for (int i = 0; i < num_pools; i++) {
    pool_head[i] = 0;
    atomic_store(&(task_pool_lock[i]), false);
  }

  /* ---------- initOctree: global init ---------- */
  atomic_store(treeSize, 100);
  atomic_store(particlesDone, 0);

  /* create and enqueue the first task */
  t->treepos = 0;
  t->middle.x = 0;
  t->middle.y = 0;
  t->middle.z = 0;
  t->middle.w = 256;

  t->beg = 0;
  t->end = numParticles;
  t->flip = false;

  /* push in pool_id zero */
  wgm_task_push(t, pools, task_pool_lock, pool_head, pool_size, 0);
  /* ---------- end of initOctree ---------- */
}

/*---------------------------------------------------------------------------*/

void octree_main(
    /* octree args */
    __global float4 *particles, __global float4 *newparticles,
    __global unsigned int *tree, const uint numParticles,
    __global atomic_uint *treeSize, __global atomic_uint *particlesDone,
    const unsigned int maxchilds, __global Task *pools,
    __global atomic_int *task_pool_lock, __global int *pool_head,
    const int num_pools, const int pool_size, __global float4 *frompart,
    __global float4 *topart, __local uint *count, __local uint *sum,
    __local Task *t, __local int *got_new_task, __local Task *newTask,
    __global IW_barrier *__bar, __global Kernel_ctx *__k_ctx,
    CL_Scheduler_ctx __s_ctx, __local int *__scratchpad,
    Restoration_ctx *__restoration_ctx) {
  ;
  ;
  ;
  ;
  ;

  /* ADD INIT HERE */
  if (__restoration_ctx->target != 0) {
  } else {
    if (k_get_global_id(__k_ctx) == 0) {
      octree_init(pools, task_pool_lock, pool_head, num_pools, pool_size,
                  treeSize, particlesDone, numParticles, &(t[0]));
    }
    global_barrier(__bar, __k_ctx);

    /* main loop */
  }
  // Hand hack: impose a non-trivial true
  while (atomic_load(particlesDone) != 1000000000) {
      /* __restoration_ctx->target != */
      /* UCHAR_MAX /\* substitute for 'true', which can cause compiler hangs *\/) { */
    switch (__restoration_ctx->target) {
    case 0:
      if (!(true)) {
        return;
      }

      /* only the first group can fork, to limit calls to offer_fork. */
      if (k_get_group_id(__k_ctx) == 0) {
        {
          Restoration_ctx __to_fork;
          __to_fork.target = 1;
          int __junk_private;
          offer_fork(__k_ctx, __s_ctx, __scratchpad, &__to_fork,
                     &__junk_private, &__junk_global);
        }
      case 1:
        __restoration_ctx->target = 0;
      }

      // can be killed before handling a task, but always keep at least
      // one work-group alive.
      if (k_get_group_id(__k_ctx) > 0) {
        offer_kill(__k_ctx, __s_ctx, __scratchpad, k_get_group_id(__k_ctx));
      }

      uint local_id = get_local_id(0);
      uint local_size = get_local_size(0);
      int pool_id = k_get_group_id(__k_ctx) % num_pools;

      // Try to acquire new task
      if (local_id == 0) {
        // First try own task, then check neighbours
        for (int i = 0; i < num_pools; i++) {
          got_new_task[0] =
              wgm_task_pop(&(t[0]), pools, task_pool_lock, pool_head, pool_size,
                           (pool_id + i) % num_pools);
          if (got_new_task[0]) {
            break;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      if (!got_new_task[0]) {
        if (atomic_load(particlesDone) >= numParticles) {
          return;
        } else {
          continue;
        }
      }

      // Process task

      if ((t[0]).flip) {
        frompart = newparticles;
        topart = particles;
      } else {
        frompart = particles;
        topart = newparticles;
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int i = local_id; i < 8; i += local_size) {
        count[i] = 0;
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int i = (t[0]).beg + local_id; i < (t[0]).end; i += local_size) {
        int box = whichbox(frompart[i], (t[0]).middle);
        atomic_add(&(count[box]), 1);
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      if (local_id == 0) {
        sum[0] = count[0];
        for (int x = 1; x < 8; x++) {
          sum[x] = sum[x - 1] + count[x];
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for (uint i = (t[0]).beg + local_id; i < (t[0]).end; i += local_size) {
        int toidx = (t[0]).beg +
                    atomic_dec(&(sum[whichbox(frompart[i], (t[0]).middle)])) -
                    1;
        topart[toidx] = frompart[i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      for (int i = 0; i < 8; i++) {

        /* Create new work or move to correct side */
        if (count[i] > maxchilds) {
          if (local_id == 0) {
            newTask[0].middle.x = (t[0]).middle.x + (t[0]).middle.w * mc[i][0];
            newTask[0].middle.y = (t[0]).middle.y + (t[0]).middle.w * mc[i][1];
            newTask[0].middle.z = (t[0]).middle.z + (t[0]).middle.w * mc[i][2];
            newTask[0].middle.w = (t[0]).middle.w / 2.0;

            newTask[0].flip = !(t[0]).flip;
            newTask[0].beg = (t[0]).beg + sum[i];
            newTask[0].end = newTask[0].beg + count[i];

            tree[(t[0]).treepos + i] = atomic_fetch_add(treeSize, (uint)8);
            newTask[0].treepos = tree[(t[0]).treepos + i];

            int pushed = false;
            for (int j = 0; j < num_pools; j++) {
              pushed =
                  wgm_task_push(&(newTask[0]), pools, task_pool_lock, pool_head,
                                pool_size, (pool_id + j) % num_pools);
              if (pushed) {
                break;
              }
            }
            if (pushed == false) {
              /* pool overflow */
              atomic_store(particlesDone, numParticles);
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        } else {
          if (!(t[0]).flip) {
            for (int j = (t[0]).beg + sum[i] + local_id;
                 j < (t[0]).beg + sum[i] + count[i]; j += local_size) {
              particles[j] = topart[j];
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (local_id == 0) {
            atomic_fetch_add(particlesDone, count[i]);
            uint val = count[i];
            tree[(t[0]).treepos + i] = 0x80000000 | val;
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      }
    }
  } // end of main loop
}

/*---------------------------------------------------------------------------*/

kernel void
mega_kernel(__global int *A, const int A_row, const int A_col, __global int *B,
            const int B_row, const int B_col, __global int *C,
            __global atomic_int *counter, __global atomic_int *hash,
            __global float4 *particles, __global float4 *newparticles,
            __global unsigned int *tree, const uint numParticles,
            __global atomic_uint *treeSize, __global atomic_uint *particlesDone,
            const unsigned int maxchilds, __global Task *pools,
            __global atomic_int *task_pool_lock, __global int *pool_head,
            const int num_pools, const int pool_size, __global float4 *frompart,
            __global float4 *topart, __global IW_barrier *bar,
            __global Discovery_ctx *d_ctx,
            __global Kernel_ctx *non_persistent_kernel_ctx,
            __global Kernel_ctx *persistent_kernel_ctx, SCHEDULER_ARGS) {
  __local uint count[8];
  __local uint sum[8];
  __local Task t[1];
  __local int got_new_task[1];
  __local Task newTask[1];
#define NON_PERSISTENT_KERNEL                                                  \
  matmult(A, A_row, A_col, B, B_row, B_col, C, counter, hash,                  \
          non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  octree_main(particles, newparticles, tree, numParticles, treeSize,           \
              particlesDone, maxchilds, pools, task_pool_lock, pool_head,      \
              num_pools, pool_size, frompart, topart, count, sum, t,           \
              got_new_task, newTask, bar, persistent_kernel_ctx, s_ctx,        \
              scratchpad, &r_ctx_local)
#include "main_device_body.cl"
}
//
