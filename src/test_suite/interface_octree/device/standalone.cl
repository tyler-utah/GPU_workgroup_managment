#include "../rt_common/cl_types.h"
#include "restoration_ctx.h"
#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

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

int myrand(__global int *randdata, int group_id) {
  /* Hugues: size of randdata is set in host code, it is considered to
     be an upper bound to the possible number of groups */
  randdata[group_id] = randdata[group_id] * 1103515245 + 12345;
  return ((unsigned)(randdata[group_id] / 65536) % 32768) + group_id;
}

/*---------------------------------------------------------------------------*/
/* lbabp: load balance ABP style, aka work-stealing */

void DLBABP_push(__global Task *deq, __global DequeHeader *dh,
                 unsigned int maxlength, __local Task *val, int group_id) {
  int private_tail = atomic_load_explicit(
      &(dh[group_id].tail), memory_order_acquire, memory_scope_device);
  deq[group_id * maxlength + private_tail] = *val;
  private_tail++;
  atomic_store_explicit(&(dh[group_id].tail), private_tail,
                        memory_order_release, memory_scope_device);
}

/*---------------------------------------------------------------------------*/

void DLBABP_enqueue(__global Task *deq, __global DequeHeader *dh,
                    unsigned int maxlength, __local Task *val, int group_id) {
  /* Hugues todo: check calls to DLBABP_enqueue, can any other thread
   * than id0 can call it ? */
  if (get_local_id(0) == 0) {
    DLBABP_push(deq, dh, maxlength, val, group_id);
  }
}

/*---------------------------------------------------------------------------*/

/* Hugues: head is separated in ctr and index, ctr is useful to avoid
 * ABA problem. Since CAS operation deals with 32 bits int, a head is
 * declared as an int, and ctr/index is accessed with mask and logical
 * AND operations. */

#define getIndex(head) (head & 0xffff)

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
  /* IncIndex */
  newHead = oldHead + 1;
  if (atomic_compare_exchange_strong_explicit(
          &(dh[idx].head), &oldHead, newHead, memory_order_acq_rel,
          memory_order_relaxed, memory_scope_device)) {
    return 1;
  }

  return -1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_pop(__global Task *deq, __global DequeHeader *dh,
               unsigned int maxlength, __local Task *val, int group_id) {
  int localTail;
  int oldHead;
  int newHead;

  localTail = atomic_load_explicit(&(dh[group_id].tail), memory_order_acquire,
                                   memory_scope_device);
  if (localTail == 0) {
    return -1;
  }

  localTail--;

  atomic_store_explicit(&(dh[group_id].tail), localTail, memory_order_release,
                        memory_scope_device);

  *val = deq[group_id * maxlength + localTail];

  oldHead = atomic_load_explicit(&(dh[group_id].head), memory_order_acquire,
                                 memory_scope_device);

  if (localTail > getIndex(oldHead)) {
    return 1;
  }

  atomic_store_explicit(&(dh[group_id].tail), 0, memory_order_release,
                        memory_scope_device);
  /* Hugues: inline getZeroIndexIncCtr below */
  newHead = (oldHead + 0x10000) & 0xffff0000;
  if (localTail == getIndex(oldHead)) {
    if (atomic_compare_exchange_strong_explicit(
            &(dh[group_id].head), &oldHead, newHead, memory_order_acq_rel,
            memory_order_relaxed, memory_scope_device)) {
      return 1;
    }
  }
  atomic_store_explicit(&(dh[group_id].head), newHead, memory_order_release,
                        memory_scope_device);
  return -1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue2(__global Task *deq, __global DequeHeader *dh,
                    unsigned int maxlength, __local Task *val,
                    __global int *randdata, int num_pools, int group_id) {
  if (DLBABP_pop(deq, dh, maxlength, val, group_id) == 1) {
    return 1;
  }

  if (DLBABP_steal(deq, dh, maxlength, val,
                   myrand(randdata, group_id) % num_pools) == 1) {
    return 1;
  } else {
    return 0;
  }
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue(__global Task *deq, __global DequeHeader *dh,
                   unsigned int maxlength, __local Task *val,
                   __global int *randdata, int num_pools,
                   __local volatile int *rval, int group_id) {
  int dval = 0;

  if (get_local_id(0) == 0) {
    *rval =
        DLBABP_dequeue2(deq, dh, maxlength, val, randdata, num_pools, group_id);
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

void octree_init(__global Task *deq, __global DequeHeader *dh,
                 unsigned int maxlength, __global unsigned int *treeSize,
                 __global unsigned int *particlesDone, const int num_pools,
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

  /* create and enqueue the first task */
  t->treepos = 0;
  t->middle.x = 0;
  t->middle.y = 0;
  t->middle.z = 0;
  t->middle.w = 256;

  t->beg = 0;
  t->end = numParticles;
  t->flip = false;

  DLBABP_enqueue(deq, dh, maxlength, t, 0);
  /* ---------- end of initOctree ---------- */
}

/*---------------------------------------------------------------------------*/

__kernel void octree_main(
    /* octree args */
    __global int *randdata, __global float4 *particles,
    __global float4 *newparticles, __global unsigned int *tree,
    const unsigned int numParticles, __global unsigned int *treeSize,
    __global unsigned int *particlesDone, const unsigned int maxchilds,
    const int num_pools, __global Task *deq, __global DequeHeader *dh,
    const unsigned int maxlength, __global float4 *frompart,
    __global float4 *topart, __global IW_barrier *__bar,
    __global Discovery_ctx *__d_ctx, SCHEDULER_ARGS) {
  /* Hugues: pointers to global memory, but the pointers are stored in
     local memory */
  __local int __scratchpad[2];
  DISCOVERY_PROTOCOL(__d_ctx, __scratchpad);
  INIT_SCHEDULER;
  __local unsigned int count[8];
  __local int sum[8];

  __local Task t[1];
  __local unsigned int check[1];

  __local Task newTask[1];

  __local volatile int rval[1];

  /* ADD INIT HERE */
  if (p_get_group_id(__d_ctx) == 0) {
    if (get_local_id(0) == 0) {
      atomic_store_explicit(s_ctx.persistent_flag, __d_ctx->count,
                            memory_order_release, memory_scope_all_svm_devices);
      atomic_store_explicit(s_ctx.scheduler_flag, DEVICE_WAITING,
                            memory_order_release, memory_scope_all_svm_devices);
      while (atomic_load_explicit(s_ctx.scheduler_flag, memory_order_acquire,
                                  memory_scope_all_svm_devices) !=
             DEVICE_TO_PERSISTENT_TASK)
        ;
    }
    BARRIER;
  }
  global_barrier_disc(__bar, __d_ctx);
  if (get_local_id(0) == 0) {
    if (p_get_global_id(__d_ctx) == 0) {
      octree_init(deq, dh, maxlength, treeSize, particlesDone, num_pools,
                  numParticles, t);
    }
  }
  global_barrier_disc(__bar, __d_ctx);

  /* main loop */
  while (true) {

    uint local_id = get_local_id(0);
    uint local_size = get_local_size(0);

    // only the first group can fork, to limit calls to offer_fork.
    if (p_get_group_id(__d_ctx) == 0) {
      ;
    }

    // can be killed before handling a task, but always keep at least
    // one work-group alive.
    if (p_get_group_id(__d_ctx) > 0) {
      ;
    }

    int group_id = p_get_group_id(__d_ctx);

    // Try to acquire new task
    if (DLBABP_dequeue(deq, dh, maxlength, &(t[0]), randdata, num_pools, rval,
                       group_id) == 0) {
      (check[0]) = *particlesDone;
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((check[0]) == numParticles) {
        break;
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
          DLBABP_enqueue(deq, dh, maxlength, &newTask[0], group_id);
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
  if (get_local_id(0) == 0) {
    atomic_fetch_sub_explicit(s_ctx.persistent_flag, 1, memory_order_acq_rel,
                              memory_scope_all_svm_devices);
  }
  // end of main loop
}
//