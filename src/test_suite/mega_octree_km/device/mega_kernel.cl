#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

__global int __junk_global;

void MY_reduce(int length, __global int *buffer, __global atomic_int *result,
               __local int *scratch, __global Kernel_ctx *__k_ctx) {

  ;
  int gid = k_get_global_id(__k_ctx);
  int local_index = get_local_id(0);
  int stride = k_get_global_size(__k_ctx);

  for (int global_index = gid; global_index < length; global_index += stride) {
    if (global_index < length) {
      scratch[local_index] = buffer[global_index];
    } else {
      scratch[local_index] = INT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 1; offset < get_local_size(0); offset <<= 1) {
      int mask = (offset << 1) - 1;
      if ((local_index & mask) == 0) {
        int other = scratch[local_index + offset];
        int mine = scratch[local_index];
        scratch[local_index] = (mine < other) ? mine : other;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
      atomic_fetch_min((result), scratch[0]);
    }
  }
}

__global int __junk_global;

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

int myrand(__global Kernel_ctx *__k_ctx, __global int *randdata) {
  /* Hugues: size of randdata is set in host code, it is considered to
     be an upper bound to the possible number of groups */
  int id = k_get_group_id(__k_ctx);
  randdata[id] = randdata[id] * 1103515245 + 12345;
  return ((unsigned)(randdata[id] / 65536) % 32768) + id;
}

/*---------------------------------------------------------------------------*/
/* lbabp: load balance ABP style, aka work-stealing */

void DLBABP_push(__global Kernel_ctx *__k_ctx, __global Task *deq,
                 __global DequeHeader *dh, unsigned int maxlength,
                 __local Task *val, __global volatile int *maxl) {
  int id = k_get_group_id(__k_ctx);
  int private_tail = atomic_load_explicit(&(dh[id].tail), memory_order_acquire,
                                          memory_scope_device);
  deq[id * maxlength + private_tail] = *val;
  private_tail++;
  atomic_store_explicit(&(dh[id].tail), private_tail, memory_order_release,
                        memory_scope_device);

  if (*maxl < private_tail) {
    atomic_max(maxl, private_tail);
  }
}

/*---------------------------------------------------------------------------*/

void DLBABP_enqueue(__global Kernel_ctx *__k_ctx, __global Task *deq,
                    __global DequeHeader *dh, unsigned int maxlength,
                    __local Task *val, __global volatile int *maxl) {
  /* Hugues todo: check calls to DLBABP_enqueue, can any other thread
   * than id0 can call it ? */
  if (get_local_id(0) == 0) {
    DLBABP_push(__k_ctx, deq, dh, maxlength, val, maxl);
  }
}

/*---------------------------------------------------------------------------*/

/* Hugues: head is separated in ctr and index, ctr is useful to avoid
 * ABA problem. Since CAS operation deals with 32 bits int, a head is
 * declared as an int, and ctr/index is accessed with mask and logical
 * AND operations. */

int getIndex(int head) { return head & 0xffff; }

/*---------------------------------------------------------------------------*/

int getZeroIndexIncCtr(int head) { return (head + 0x10000) & 0xffff0000; }

/*---------------------------------------------------------------------------*/

int incIndex(int head) { return head + 1; }

/*---------------------------------------------------------------------------*/

int DLBABP_steal(__global Task *deq, __global DequeHeader *dh,
                 unsigned int maxlength, __local Task *val, unsigned int idx) {
  int remoteTail;
  int oldHead;
  int newHead;

  oldHead = atomic_load_explicit(&(dh[idx].head), memory_order_acquire,
                                 memory_scope_device);
  /* We need to access dh[idx].tail but we do not modify it,
     therefore a single load-acquire is enough */
  remoteTail = atomic_load_explicit(&(dh[idx].tail), memory_order_acquire,
                                    memory_scope_device);
  if (remoteTail <= getIndex(oldHead)) {
    return -1;
  }

  *val = deq[idx * maxlength + getIndex(oldHead)];
  newHead = incIndex(oldHead);
  if (atomic_compare_exchange_strong_explicit(
          &(dh[idx].head), &oldHead, newHead, memory_order_acq_rel,
          memory_order_relaxed, memory_scope_device)) {
    return 1;
  }

  return -1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_pop(__global Kernel_ctx *__k_ctx, __global Task *deq,
               __global DequeHeader *dh, unsigned int maxlength,
               __local Task *val) {
  int localTail;
  int oldHead;
  int newHead;
  int id = k_get_group_id(__k_ctx);

  localTail = atomic_load_explicit(&(dh[id].tail), memory_order_acquire,
                                   memory_scope_device);
  if (localTail == 0) {
    return -1;
  }

  localTail--;

  atomic_store_explicit(&(dh[id].tail), localTail, memory_order_release,
                        memory_scope_device);

  *val = deq[id * maxlength + localTail];

  oldHead = atomic_load_explicit(&(dh[id].head), memory_order_acquire,
                                 memory_scope_device);

  if (localTail > getIndex(oldHead)) {
    return 1;
  }

  atomic_store_explicit(&(dh[id].tail), 0, memory_order_release,
                        memory_scope_device);
  newHead = getZeroIndexIncCtr(oldHead);
  if (localTail == getIndex(oldHead)) {
    if (atomic_compare_exchange_strong_explicit(
            &(dh[id].head), &oldHead, newHead, memory_order_acq_rel,
            memory_order_relaxed, memory_scope_device)) {
      return 1;
    }
  }
  atomic_store_explicit(&(dh[id].head), newHead, memory_order_release,
                        memory_scope_device);
  return -1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue2(__global Kernel_ctx *__k_ctx, __global Task *deq,
                    __global DequeHeader *dh, unsigned int maxlength,
                    __local Task *val, __global int *randdata,
                    unsigned int *localStealAttempts, int num_pools) {
  if (DLBABP_pop(__k_ctx, deq, dh, maxlength, val) == 1) {
    return 1;
  }

  *localStealAttempts += 1;

  if (DLBABP_steal(deq, dh, maxlength, val,
                   myrand(__k_ctx, randdata) % num_pools) == 1) {
    return 1;
  } else {
    return 0;
  }
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue(__global Kernel_ctx *__k_ctx, __global Task *deq,
                   __global DequeHeader *dh, unsigned int maxlength,
                   __local Task *val, __global int *randdata,
                   unsigned int *localStealAttempts, int num_pools,
                   __local volatile int *rval) {
  int dval = 0;

  if (get_local_id(0) == 0) {
    *rval = DLBABP_dequeue2(__k_ctx, deq, dh, maxlength, val, randdata,
                            localStealAttempts, num_pools);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  dval = *rval;
  barrier(CLK_LOCAL_MEM_FENCE);

  return dval;
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

void octree_init(__global Kernel_ctx *__k_ctx, __global Task *deq,
                 __global DequeHeader *dh, unsigned int maxlength,
                 __global unsigned int *treeSize,
                 __global unsigned int *particlesDone,
                 __global volatile int *maxl,
                 __global unsigned int *stealAttempts, const int num_pools,
                 unsigned int numParticles, __local Task *t) {
  /* reset queues */
  int i;
  for (i = 0; i < num_pools; i++) {
    atomic_store(&(dh[i].head), 0);
    atomic_store(&(dh[i].tail), 0);
  }

  /* ---------- initOctree: global init ---------- */
  *treeSize = 100;
  *particlesDone = 0;
  /* In Cuda, maxl is a kernel global initialized to 0 */
  *maxl = 0;
  *stealAttempts = 0;

  /* create and enqueue the first task */
  t->treepos = 0;
  t->middle.x = 0;
  t->middle.y = 0;
  t->middle.z = 0;
  t->middle.w = 256;

  t->beg = 0;
  t->end = numParticles;
  t->flip = false;

  DLBABP_enqueue(__k_ctx, deq, dh, maxlength, t, maxl);
  /* ---------- end of initOctree ---------- */
}

/*---------------------------------------------------------------------------*/

void octree_main(
    /* octree args */
    __global atomic_int *num_iterations, __global int *randdata,
    __global volatile int *maxl, __global float4 *particles,
    __global float4 *newparticles, __global unsigned int *tree,
    const unsigned int numParticles, __global unsigned int *treeSize,
    __global unsigned int *particlesDone, const unsigned int maxchilds,
    __global unsigned int *stealAttempts, const int num_pools,
    __global Task *deq, __global DequeHeader *dh, const unsigned int maxlength,
    __global float4 *frompart, __global float4 *topart,
    __local unsigned int *count, __local int *sum, __local Task *t,
    __local unsigned int *check, __local Task *newTask,
    volatile __local int *rval, __global IW_barrier *__bar,
    __global Kernel_ctx *__k_ctx, CL_Scheduler_ctx __s_ctx,
    __local int *__scratchpad, Restoration_ctx *__restoration_ctx) {
  /* Hugues: pointers to global memory, but the pointers are stored in
     local memory */
  ;
  ;

  ;
  ;

  ;

  uint local_id = get_local_id(0);
  uint local_size = get_local_size(0);

  uint localStealAttempts;

  ;

  int i;

  /* ADD INIT HERE */
  if (__restoration_ctx->target != 0) {
    local_id = __restoration_ctx->local_id;
    local_size = __restoration_ctx->local_size;
    localStealAttempts = __restoration_ctx->localStealAttempts;
    i = __restoration_ctx->i;
  } else {
    if (get_local_id(0) == 0) {
      localStealAttempts = 0;
      if (k_get_global_id(__k_ctx) == 0) {
        octree_init(__k_ctx, deq, dh, maxlength, treeSize, particlesDone, maxl,
                    stealAttempts, num_pools, numParticles, t);
      }
    }
    global_barrier(__bar, __k_ctx);

    /* main loop */
  }
  while (
      __restoration_ctx->target !=
      UCHAR_MAX /* substitute for 'true', which can cause compiler hangs */) {
    switch (__restoration_ctx->target) {
    case 0:
      if (!(true)) {
        return;
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      /* can be killed before handling a task, but always keep at least
         one work-group alive. This is to avoid to call octree_init()
         after a cfork() */
      if (k_get_group_id(__k_ctx) > 0) {
        offer_kill(__k_ctx, __s_ctx, __scratchpad, k_get_group_id(__k_ctx));
      }

      // always suggest to fork

      /* Hugues: variable 'i' is just used to give a valid argument, we */
      /* do not use the returned value. */

      /* Hugues: the octree_bar->num_groups arg is here to put something */
      /* valid as argument, I don't think this value is used anywhere */
      /* else. I jus mimick the call to cfork() in */
      /* global_barrier_resize(). But looking at the code of */
      /* global_barrier(), bar->num_groups is not used there. */

      if (k_get_group_id(__k_ctx) == 0) {
        {
          Restoration_ctx __to_fork;
          __to_fork.target = 1;
          __to_fork.local_id = local_id;
          __to_fork.local_size = local_size;
          __to_fork.localStealAttempts = localStealAttempts;
          /* __to_fork.i = i; */
          /* int __junk_private; */
          /* offer_fork(__k_ctx, __s_ctx, __scratchpad, &__to_fork, */
          /*            &__junk_private, &__junk_global); */
        }
      case 1:
        __restoration_ctx->target = 0;
      }

      // Try to acquire new task
      if (DLBABP_dequeue(__k_ctx, deq, dh, maxlength, &(t[0]), randdata,
                         &localStealAttempts, num_pools, rval) == 0) {
        (check[0]) = *particlesDone;
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((check[0]) == numParticles) {
          if (local_id == 0) {
            atomic_add(stealAttempts, localStealAttempts);
          }
          return;
        }
        continue;
      }

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
        /* Hugues todo: use atomic_inc() here ? */
        atomic_add(&count[whichbox(frompart[i], (t[0]).middle)], 1);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      if (local_id == 0) {
        sum[0] = count[0];
        for (int x = 1; x < 8; x++)
          sum[x] = sum[x - 1] + count[x];
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for (unsigned int i = (t[0]).beg + local_id; i < (t[0]).end;
           i += local_size) {
        /* Hugues: use atomic_dec() here ? */
        int toidx = (t[0]).beg +
                    atomic_add(&sum[whichbox(frompart[i], (t[0]).middle)], -1) -
                    1;
        topart[toidx] = frompart[i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      /* Hugues: todo: i+= 1 ---> i++ */
      for (int i = 0; i < 8; i += 1) {

        // Create new work or move to correct side
        if (count[i] > maxchilds) {
          if (local_id == 0) {
            newTask[0].middle.x = (t[0]).middle.x + (t[0]).middle.w * mc[i][0];
            newTask[0].middle.y = (t[0]).middle.y + (t[0]).middle.w * mc[i][1];
            newTask[0].middle.z = (t[0]).middle.z + (t[0]).middle.w * mc[i][2];
            newTask[0].middle.w = (t[0]).middle.w / 2.0;

            newTask[0].flip = !(t[0]).flip;
            newTask[0].beg = (t[0]).beg + sum[i];
            newTask[0].end = newTask[0].beg + count[i];

            tree[(t[0]).treepos + i] = atomic_add(treeSize, (unsigned int)8);
            newTask[0].treepos = tree[(t[0]).treepos + i];
            DLBABP_enqueue(__k_ctx, deq, dh, maxlength, &newTask[0], maxl);
          }
        } else {
          if (!(t[0]).flip) {
            for (int j = (t[0]).beg + sum[i] + local_id;
                 j < (t[0]).beg + sum[i] + count[i]; j += local_size) {
              particles[j] = topart[j];
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE);
          if (local_id == 0) {
            atomic_add(particlesDone, count[i]);
            unsigned int val = count[i];
            tree[(t[0]).treepos + i] = 0x80000000 | val;
          }
        }
      }
    }
  } // end of main loop
}

kernel void
mega_kernel(int length, __global int *buffer, __global atomic_int *result,
            __global atomic_int *num_iterations, __global int *randdata,
            __global volatile int *maxl, __global float4 *particles,
            __global float4 *newparticles, __global unsigned int *tree,
            const unsigned int numParticles, __global unsigned int *treeSize,
            __global unsigned int *particlesDone, const unsigned int maxchilds,
            __global unsigned int *stealAttempts, const int num_pools,
            __global Task *deq, __global DequeHeader *dh,
            const unsigned int maxlength, __global float4 *frompart,
            __global float4 *topart, __global IW_barrier *bar,
            __global Discovery_ctx *d_ctx,
            __global Kernel_ctx *non_persistent_kernel_ctx,
            __global Kernel_ctx *persistent_kernel_ctx, SCHEDULER_ARGS) {
  __local int scratch[256];
  __local unsigned int count[8];
  __local int sum[8];
  __local Task t[1];
  __local unsigned int check[1];
  __local Task newTask[1];
  __local volatile int rval[1];
#define NON_PERSISTENT_KERNEL                                                  \
  MY_reduce(length, buffer, result, scratch, non_persistent_kernel_ctx)
#define PERSISTENT_KERNEL                                                      \
  octree_main(num_iterations, randdata, maxl, particles, newparticles, tree,   \
              numParticles, treeSize, particlesDone, maxchilds, stealAttempts, \
              num_pools, deq, dh, maxlength, frompart, topart, count, sum, t,  \
              check, newTask, rval, bar, persistent_kernel_ctx, s_ctx,         \
              scratchpad, &r_ctx_local)
#include "main_device_body.cl"
}
//
