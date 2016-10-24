// It is important to include this first because other files use it.
#include "restoration_ctx.h"

#include "discovery.cl"
#include "kernel_ctx.cl"
#include "cl_scheduler.cl"
#include "iw_barrier.cl"

/*---------------------------------------------------------------------------*/

// This is the "graphics kernel"
void MY_reduce(int length,
               __global int* buffer,
               __global atomic_int* result,

               // New arg, the kernel ctx
               __global Kernel_ctx *kernel_ctx)
{
  __local int scratch[256];
  int gid = k_get_global_id(kernel_ctx);
  int stride = k_get_global_size(kernel_ctx);
  int local_index = get_local_id(0);

  for (int global_index = gid; global_index < length; global_index += stride) {

    if (global_index < length) {
      scratch[local_index] = buffer[global_index];
    } else {
      scratch[local_index] = INT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = 1;
        offset < get_local_size(0);
        offset <<= 1) {
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

/* ========================================================================= */
// Persistent Kernel -- octree

/* Note: many names refer to DLBABP, this comes from the CUDA version,
   and it stands for "Dynamic Load Balancing - Arora Blumofe Plaxton"
   (authors of work-stealing paper). There used to be a DLBABP
   class/structure which contained deq, dh and maxlength, but we've got
   rid of it since its initialisation required to edit pointers on the
   device side. So now we just pass deq, dh and maxlength around. */

typedef struct {
  float4 middle;
  bool flip;
  unsigned int end;
  unsigned int beg;
  unsigned int treepos;
} Task;

/*---------------------------------------------------------------------------*/

typedef struct {
  atomic_int tail;
  atomic_int head;
} DequeHeader;

/*---------------------------------------------------------------------------*/
/* rand */

int myrand(__global Kernel_ctx *kernel_ctx, __global int *randdata) {
  /* Hugues: size of randdata is set in host code, it is considered to
     be an upper bound to the possible number of groups */
  int id = k_get_group_id(kernel_ctx);
  randdata[id] = randdata[id] * 1103515245 + 12345;
  return((unsigned)(randdata[id] / 65536) % 32768) + id;
}

/*---------------------------------------------------------------------------*/
/* lbabp: load balance ABP style, aka work-stealing */

void DLBABP_push(__global Kernel_ctx *kernel_ctx, __global Task *deq, __global DequeHeader *dh, unsigned int maxlength, __local Task *val, __global volatile int *maxl) {
  int id = k_get_group_id(kernel_ctx);
  int private_tail = atomic_load_explicit(&(dh[id].tail), memory_order_acquire, memory_scope_device);
  deq[id * maxlength + private_tail] = *val;
  private_tail++;
  atomic_store_explicit(&(dh[id].tail), private_tail, memory_order_release, memory_scope_device);

  if (*maxl < private_tail) {
    atomic_max(maxl, private_tail);
  }
}

/*---------------------------------------------------------------------------*/

void DLBABP_enqueue(__global Kernel_ctx *kernel_ctx, __global Task *deq, __global DequeHeader *dh, unsigned int maxlength, __local Task *val, __global volatile int *maxl) {
  /* Hugues todo: check calls to DLBABP_enqueue, can any other thread
   * than id0 can call it ? */
  if (get_local_id(0) == 0) {
    DLBABP_push(kernel_ctx, deq, dh, maxlength, val, maxl);
  }
}

/*---------------------------------------------------------------------------*/

/* Hugues: head is separated in ctr and index, ctr is useful to avoid
 * ABA problem. Since CAS operation deals with 32 bits int, a head is
 * declared as an int, and ctr/index is accessed with mask and logical
 * AND operations. */

int getIndex(int head) {
  return head & 0xffff;
}

/*---------------------------------------------------------------------------*/

int getZeroIndexIncCtr(int head) {
  return (head + 0x10000) & 0xffff0000;
}

/*---------------------------------------------------------------------------*/

int incIndex(int head) {
  return head + 1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_steal(__global Task *deq, __global DequeHeader *dh, unsigned int maxlength,  __local Task *val, unsigned int idx) {
  int remoteTail;
  int oldHead;
  int newHead;

  oldHead = atomic_load_explicit(&(dh[idx].head), memory_order_acquire, memory_scope_device);
  /* We need to access dh[idx].tail but we do not modify it,
     therefore a single load-acquire is enough */
  remoteTail = atomic_load_explicit(&(dh[idx].tail), memory_order_acquire, memory_scope_device);
  if(remoteTail <= getIndex(oldHead)) {
    return -1;
  }

  *val = deq[idx * maxlength + getIndex(oldHead)];
  newHead = incIndex(oldHead);
  if (atomic_compare_exchange_weak_explicit(&(dh[idx].head), &oldHead, newHead, memory_order_acq_rel, memory_order_relaxed, memory_scope_device)) {
    return 1;
  }

  return -1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_pop(__global Kernel_ctx *kernel_ctx,  __global Task *deq, __global DequeHeader *dh, unsigned int maxlength, __local Task *val) {
  int localTail;
  int oldHead;
  int newHead;
  int id = k_get_group_id(kernel_ctx);

  localTail = atomic_load_explicit(&(dh[id].tail), memory_order_acquire, memory_scope_device);
  if(localTail == 0) {
    return -1;
  }

  localTail--;

  atomic_store_explicit(&(dh[id].tail), localTail, memory_order_release, memory_scope_device);

  *val = deq[id * maxlength + localTail];

  oldHead = atomic_load_explicit(&(dh[id].head), memory_order_acquire, memory_scope_device);

  if (localTail > getIndex(oldHead)) {
    return 1;
  }

  atomic_store_explicit(&(dh[id].tail), 0, memory_order_release, memory_scope_device);
  newHead = getZeroIndexIncCtr(oldHead);
  if(localTail == getIndex(oldHead)) {
    if(atomic_compare_exchange_weak_explicit(&(dh[id].head), &oldHead, newHead, memory_order_acq_rel, memory_order_release, memory_scope_device)) {
      return 1;
    }
  }
  atomic_store_explicit(&(dh[id].head), newHead, memory_order_release, memory_scope_device);
  return -1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue2(__global Kernel_ctx *kernel_ctx, __global Task *deq, __global DequeHeader *dh, unsigned int maxlength,  __local Task *val, __global int *randdata, unsigned int *localStealAttempts, int num_pools)
{
  if (DLBABP_pop(kernel_ctx, deq, dh, maxlength, val) == 1) {
    return 1;
  }

  *localStealAttempts += 1;

  if (DLBABP_steal(deq, dh, maxlength, val, myrand(kernel_ctx, randdata) % num_pools) == 1) {
    return 1;
  } else {
    return 0;
  }
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue(__global Kernel_ctx *kernel_ctx, __global Task *deq, __global DequeHeader *dh, unsigned int maxlength, __local Task *val, __global int *randdata, unsigned int *localStealAttempts, int num_pools) {
  __local volatile int rval;
  int dval = 0;

  if(get_local_id(0) == 0) {
    rval = DLBABP_dequeue2(kernel_ctx, deq, dh, maxlength, val, randdata, localStealAttempts, num_pools);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  dval = rval;
  barrier(CLK_LOCAL_MEM_FENCE);

  return dval;
}

/*---------------------------------------------------------------------------*/

__constant int mc[8][3] =
{{-1,-1,-1},{+1,-1,-1},{-1,+1,-1},{+1,+1,-1},
 {-1,-1,+1},{+1,-1,+1},{-1,+1,+1},{+1,+1,+1}};

/*---------------------------------------------------------------------------*/

int whichbox(volatile float4 pos, float4 middle)
{
	int box = pos.x < middle.x ? 0 : 1;
	box    += pos.y < middle.y ? 0 : 2;
	box    += pos.z < middle.z ? 0 : 4;
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

void octree_init(
                 __global Kernel_ctx *kernel_ctx,
                 __global Task *deq,
                 __global DequeHeader *dh,
                 unsigned int maxlength,
                 __global unsigned int* treeSize,
                 __global unsigned int* particlesDone,
                 __global volatile int *maxl,
                 __global unsigned int *stealAttempts,
                 const int num_pools,
                 unsigned int numParticles
                 )
{
  /* reset queues */
  __local Task t;
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
  t.treepos=0;
  t.middle.x=0;
  t.middle.y=0;
  t.middle.z=0;
  t.middle.w=256;

  t.beg = 0;
  t.end = numParticles;
  t.flip = false;

  DLBABP_enqueue(kernel_ctx, deq, dh, maxlength, &t, maxl);
  /* ---------- end of initOctree ---------- */
}

/*---------------------------------------------------------------------------*/

void global_barrier_sense_reversal(__global IW_barrier *bar, __local int *sense, __global Kernel_ctx *k_ctx)
{
  if (get_local_id(0) == 0) {
    *sense = !(*sense);
    while (true) {
      int bar_counter = atomic_load(&(bar->counter));
      int num_groups = k_get_num_groups(k_ctx);
      if (bar_counter == num_groups) {
        break;
      }
    }
  }

  /* if (get_local_id(0) == 0) { */
  /*   *sense = !(*sense); */
  /*   if (atomic_fetch_add(&(bar->counter), 1) == 0) { */
  /*     /\* only the first to hit the barrier enters here. it spins waiting */
  /*        for the other workgroups to arrive. The number of workgroups is */
  /*        dynamic, so it should be checked from kernel_ctx everytime *\/ */
  /*     //while (true) { */
  /*       /\* Here we MUST first load the barrier counter. If not, the */
  /*          following can happen: load the number of groups, say it's */
  /*          equal to n. Then, concurrently, the scheduler allocates a new */
  /*          group, so now the number of groups is (n+1), and n groups */
  /*          enter the barrier. Now the barrier count is loaded: it is */
  /*          equal to n, therefore the barrier will release everybody, */
  /*          although it must have waited for (n+1) groups ! *\/ */
  /*       //int bar_counter = atomic_load(&(bar->counter)); */
  /*       //int num_groups = k_get_num_groups(k_ctx); */
  /*       //if (bar_counter == num_groups) { */
  /*         /\* everyone is here, first reset the counter *\/ */
  /*         //atomic_store(&(bar->counter), 0); */
  /*         /\* then release everybody *\/ */
  /*         atomic_store(&(bar->sense), *sense); */
  /*         //return; */
  /*         // } */
  /*         //} */
  /*   } else { */
  /*     /\* spin on the sense flag *\/ */
  /*     while (*sense != atomic_load(&(bar->sense))) { */

  /*     } */
  /*   } */
  /* } */

  /* Here a local barrier to stop all threads of the group. Maybe we
     need to use the value of 'sense' to make this barrier effective? */
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

/*---------------------------------------------------------------------------*/

void octree_main (
                  /* mega-kernel args */
                  __global Kernel_ctx *kernel_ctx,
                  CL_Scheduler_ctx scheduler_ctx,
                  __local int *scratchpad,
                  Restoration_ctx *r_ctx,

                  /* octree args */
                  __global IW_barrier *octree_bar,
                  __global atomic_int *num_iterations,
                  __global int *randdata,
                  __global volatile int *maxl,
                  __global float4* particles,
                  __global float4* newparticles,
                  __global unsigned int* tree,
                  const unsigned int numParticles,
                  __global unsigned int* treeSize,
                  __global unsigned int* particlesDone,
                  const unsigned int maxchilds,
                  __global unsigned int *stealAttempts,
                  const int num_pools,
                  __global Task *deq,
                  __global DequeHeader *dh,
                  const unsigned int maxlength
                  )
{
  /* Hugues: pointers to global memory, but the pointers are stored in
     local memory */
  __global float4* __local frompart;
  __global float4* __local topart;

  __local unsigned int count[8];
  __local int sum[8];

  __local Task t;
  __local unsigned int check;

  unsigned int local_id = get_local_id(0);
  unsigned int local_size = get_local_size(0);

  unsigned int localStealAttempts;

  __local int num_iter;
  __local int sense;

  Restoration_ctx to_fork;
  int i;

  if (local_id == 0) {
    if (r_ctx->target == 0) {
      /* very first time */
      sense = 0;
    } else {
      /* we have been forked */
      sense = r_ctx->sense;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // global_barrier_sense_reversal(octree_bar, &sense, kernel_ctx);

  /* Hugues: do the octree partitionning several times to last longer */

  /* while (true) { */

  /*   if (local_id == 0) { */
  /*     num_iter = atomic_load(num_iterations); */
  /*   } */

  /*   barrier(CLK_LOCAL_MEM_FENCE); */
  /*   global_barrier_sense_reversal(octree_bar, &sense, kernel_ctx); */

  /*   if (num_iter == 0) { */
  /*     return; */
  /*   } */

    if (k_get_global_id(kernel_ctx) == 0) {
      //num_iter--;
      octree_init(kernel_ctx, deq, dh, maxlength, treeSize, particlesDone, maxl, stealAttempts, num_pools, numParticles);
      //atomic_store(num_iterations, num_iter);
    }

    /* if (local_id == 0) { */
    /*   localStealAttempts = 0; */
    /* } */

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    //global_barrier_sense_reversal(octree_bar, &sense, kernel_ctx);

    /* main loop */
    while (true) {

      barrier(CLK_LOCAL_MEM_FENCE);

      /* can be killed before handling a task, but always keep at least
         one work-group alive. This is to avoid to call octree_init()
         after a cfork() */
      if (k_get_group_id(kernel_ctx) > 0) {
        if (__ckill(kernel_ctx, scheduler_ctx, scratchpad, k_get_group_id(kernel_ctx)) == -1) {
          return;
        }
      }

      // always suggest to fork

      /* Hugues: variable 'i' is just used to give a valid argument, we */
      /* do not use the returned value. */

      /* Hugues: the octree_bar->num_groups arg is here to put something */
      /* valid as argument, I don't think this value is used anywhere */
      /* else. I jus mimick the call to cfork() in */
      /* global_barrier_resize(). But looking at the code of */
      /* global_barrier(), bar->num_groups is not used there. */

      /* flag to indicate we have been forked */
      to_fork.target = 1;
      /*  */
      to_fork.sense = sense;

      cfork(kernel_ctx, scheduler_ctx, scratchpad, &to_fork, &i, &(octree_bar->num_groups));

      // Try to acquire new task
      if (DLBABP_dequeue(kernel_ctx, deq, dh, maxlength, &t, randdata, &localStealAttempts, num_pools) == 0) {
        check = *particlesDone;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (check == numParticles) {
          if (local_id == 0) {
            atomic_add(stealAttempts, localStealAttempts);
          }
          break;
        }
        continue;
      }

      // synthetic work
      for (i = 0; i < 5000; i++) {
        atomic_store(scheduler_ctx.check_value, 0);
      }

      if (t.flip) {
        frompart = newparticles;
        topart = particles;
      } else {
        frompart = particles;
        topart = newparticles;
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for(int i = local_id; i < 8; i += local_size) {
        count[i] = 0;
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for(int i = t.beg + local_id; i < t.end; i += local_size) {
        /* Hugues todo: use atomic_inc() here ? */
        atomic_add(&count[whichbox(frompart[i],t.middle)], 1);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      if (local_id == 0) {
        sum[0] = count[0];
        for (int x = 1; x < 8; x++)
          sum[x] = sum[x-1] + count[x];
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for (unsigned int i = t.beg + local_id; i < t.end; i += local_size) {
        /* Hugues: use atomic_dec() here ? */
        int toidx = t.beg + atomic_add(&sum[whichbox(frompart[i],t.middle)], -1) - 1;
        topart[toidx] = frompart[i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      /* Hugues: todo: i+= 1 ---> i++ */
      for(int i = 0; i < 8; i += 1) {
        __local Task newTask;

        // Create new work or move to correct side
        if (count[i] > maxchilds) {
          if (local_id == 0) {
            newTask.middle.x = t.middle.x + t.middle.w * mc[i][0];
            newTask.middle.y = t.middle.y + t.middle.w * mc[i][1];
            newTask.middle.z = t.middle.z + t.middle.w * mc[i][2];
            newTask.middle.w = t.middle.w / 2.0;

            newTask.flip = !t.flip;
            newTask.beg = t.beg + sum[i];
            newTask.end = newTask.beg + count[i];

            tree[t.treepos + i] = atomic_add(treeSize,(unsigned int)8);
            newTask.treepos = tree[t.treepos + i];
            DLBABP_enqueue(kernel_ctx, deq, dh, maxlength, &newTask, maxl);
          }
        } else {
          if (!t.flip) {
            for (
                 int j = t.beg + sum[i] + local_id;
                 j < t.beg + sum[i] + count[i];
                 j += local_size)
              {
                particles[j] = topart[j];
              }
          }
          barrier(CLK_LOCAL_MEM_FENCE);
          if (local_id == 0) {
            atomic_add(particlesDone, count[i]);
            unsigned int val = count[i];
            tree[t.treepos + i] = 0x80000000 | val;
          }
        }
      }
    } // end of main loop

    //global_barrier_sense_reversal(octree_bar, &sense, kernel_ctx);

    //} // end of num_iterations
}

/* ========================================================================= */

__kernel void mega_kernel(
                          // Graphics kernel args
                          int graphics_length,
                          __global int * graphics_buffer,
                          __global atomic_int * graphics_result,

                          // Persistent kernel args
                          __global IW_barrier *octree_bar,
                          __global atomic_int *num_iterations,
                          __global int *randdata,
                          __global volatile int *maxl,
                          __global float4* particles,
                          __global float4* newparticles,
                          __global unsigned int* tree,
                          unsigned int numParticles,
                          __global unsigned int* treeSize,
                          __global unsigned int* particlesDone,
                          unsigned int maxchilds,
                          __global unsigned int *stealAttempts,
                          const int num_pools,
                          __global Task *deq,
                          __global DequeHeader *dh,
                          unsigned int maxlength,

                          // Barrier object
                          __global IW_barrier *bar,

                          // Discovery context
                          __global Discovery_ctx *d_ctx,

                          // Kernel context for graphics kernel
                          __global Kernel_ctx *non_persistent_kernel_ctx,

                          // Kernel context for persistent kernel
                          __global Kernel_ctx *persistent_kernel_ctx,

                          // Scheduler args need to be passed individually
                          SCHEDULER_ARGS
                          )
{

  // These need to be made by the kernel merge tool. Its the original graphics kernel with the graphics_kernel_ctx as a final arg.
  #define NON_PERSISTENT_KERNEL MY_reduce(graphics_length, graphics_buffer, graphics_result, non_persistent_kernel_ctx)
  //#define NON_PERSISTENT_KERNEL

  // This is the original persistent kernel with the bar, persistent_kernel_ctx, s_ctx, scratchpad, and (by pointer) local restoration context.
  //#define PERSISTENT_KERNEL color_persistent(row, col, node_value, color_array, stop1, stop2, max_d, num_nodes, num_edges, bar, persistent_kernel_ctx, s_ctx, scratchpad, &r_ctx_local);
#define PERSISTENT_KERNEL octree_main(persistent_kernel_ctx, s_ctx, scratchpad, &r_ctx_local, octree_bar, num_iterations, randdata, maxl, particles, newparticles, tree, numParticles, treeSize, particlesDone, maxchilds, stealAttempts, num_pools, deq, dh, maxlength);

  // Everything else is in here
#include "main_device_body.cl"
}
//
