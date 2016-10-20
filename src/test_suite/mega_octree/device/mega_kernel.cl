
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
			__global Kernel_ctx *kernel_ctx) {
				
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
// Persistent Kernel

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

typedef struct {
  /* Hugues: pointers to other buffers declared at the host side,
   * therefore __global. Moreover, for the 'dh' variable, __global is
   * required for atomic_cmpxchg() later on */
  __global Task *deq;
  __global DequeHeader* dh;
  unsigned int maxlength;
} DLBABP;

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

void DLBABP_push(__global Kernel_ctx *kernel_ctx, __global DLBABP *dlbabp, __local Task *val, __global volatile int *maxl) {
  int id = k_get_group_id(kernel_ctx);
  int private_tail = atomic_load_explicit(&(dlbabp->dh[id].tail), memory_order_acquire, memory_scope_device);
  dlbabp->deq[id * dlbabp->maxlength + private_tail] = *val;
  private_tail++;
  atomic_store_explicit(&(dlbabp->dh[id].tail), private_tail, memory_order_release, memory_scope_device);

  if (*maxl < private_tail) {
    atomic_max(maxl, private_tail);
  }
}

/*---------------------------------------------------------------------------*/

void DLBABP_enqueue(__global Kernel_ctx *kernel_ctx, __global DLBABP *dlbabp, __local Task *val, __global volatile int *maxl) {
  /* Hugues todo: check calls to DLBABP_enqueue, can any other thread
   * than id0 can call it ? */
  if (get_local_id(0) == 0) {
    DLBABP_push(kernel_ctx, dlbabp, val, maxl);
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

int DLBABP_steal(__global DLBABP *dlbabp, __local Task *val, unsigned int idx) {
  int remoteTail;
  int oldHead;
  int newHead;

  oldHead = atomic_load_explicit(&(dlbabp->dh[idx].head), memory_order_acquire, memory_scope_device);
  /* We need to access dlbabp->dh[idx].tail but we do not modify it,
     therefore a single load-acquire is enough */
  remoteTail = atomic_load_explicit(&(dlbabp->dh[idx].tail), memory_order_acquire, memory_scope_device);
  if(remoteTail <= getIndex(oldHead)) {
    return -1;
  }

  *val = dlbabp->deq[idx * dlbabp->maxlength + getIndex(oldHead)];
  newHead = incIndex(oldHead);
  if (atomic_compare_exchange_weak_explicit(&(dlbabp->dh[idx].head), &oldHead, newHead, memory_order_acq_rel, memory_order_relaxed, memory_scope_device)) {
    return 1;
  }

  return -1;
}

/*---------------------------------------------------------------------------*/

int emptyPool(__global DLBABP *dlbabp, int group_id) {
  int localTail;
  localTail = atomic_load_explicit(&(dlbabp->dh[group_id].tail), memory_order_acquire, memory_scope_device);
  if(localTail == 0) {
    return 1;
  }
  return 0;
}

/*---------------------------------------------------------------------------*/

int DLBABP_pop(__global Kernel_ctx *kernel_ctx, __global DLBABP *dlbabp, __local Task *val) {
  int localTail;
  int oldHead;
  int newHead;
  int id = k_get_group_id(kernel_ctx);

  localTail = atomic_load_explicit(&(dlbabp->dh[id].tail), memory_order_acquire, memory_scope_device);
  if(localTail == 0) {
    return -1;
  }

  localTail--;

  atomic_store_explicit(&(dlbabp->dh[id].tail), localTail, memory_order_release, memory_scope_device);

  *val = dlbabp->deq[id * dlbabp->maxlength + localTail];

  oldHead = atomic_load_explicit(&(dlbabp->dh[id].head), memory_order_acquire, memory_scope_device);

  if (localTail > getIndex(oldHead)) {
    return 1;
  }

  atomic_store_explicit(&(dlbabp->dh[id].tail), 0, memory_order_release, memory_scope_device);
  newHead = getZeroIndexIncCtr(oldHead);
  if(localTail == getIndex(oldHead)) {
    if(atomic_compare_exchange_weak_explicit(&(dlbabp->dh[id].head), &oldHead, newHead, memory_order_acq_rel, memory_order_release, memory_scope_device)) {
      return 1;
    }
  }
  atomic_store_explicit(&(dlbabp->dh[id].head), newHead, memory_order_release, memory_scope_device);
  return -1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue2(__global Kernel_ctx *kernel_ctx, __global DLBABP *dlbabp, __local Task *val, __global int *randdata, unsigned int *localStealAttempts, int num_pools)
{
  if (DLBABP_pop(kernel_ctx, dlbabp, val) == 1) {
    return 1;
  }

  *localStealAttempts += 1;

  if (DLBABP_steal(dlbabp, val, myrand(kernel_ctx, randdata) % num_pools) == 1) {
    return 1;
  } else {
    return 0;
  }
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue(__global Kernel_ctx *kernel_ctx, __global DLBABP *dlbabp, __local Task *val, __global int *randdata, unsigned int *localStealAttempts, int num_pools) {
  __local volatile int rval;
  int dval = 0;

  if(get_local_id(0) == 0) {
    rval = DLBABP_dequeue2(kernel_ctx, dlbabp, val, randdata, localStealAttempts, num_pools);
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
                 __global DLBABP* dlbabp,
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
    atomic_store(&(dlbabp->dh[i].head), 0);
    atomic_store(&(dlbabp->dh[i].tail), 0);
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

  DLBABP_enqueue(kernel_ctx, dlbabp, &t, maxl);
  /* ---------- end of initOctree ---------- */
}

/*---------------------------------------------------------------------------*/

void octree_main (
                  /* mega-kernel args */
                  __global Kernel_ctx *kernel_ctx,
                  CL_Scheduler_ctx scheduler_ctx,
                  __local int *scratchpad,

                  /* octree args */
                  __global DLBABP* dlbabp,
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
                  unsigned int maxlength        
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

  //int NUM_ITERATIONS;

  unsigned int local_id = get_local_id(0);
  unsigned int local_size = get_local_size(0);

  unsigned int localStealAttempts;
  
  if (k_get_global_id(kernel_ctx) == 0) {
    /* ---------- initDLBABP ----------*/
    dlbabp->deq = deq;
    dlbabp->dh = dh;
    dlbabp->maxlength = maxlength;
    /* ---------- end of initDLBABP ----------*/

    //NUM_ITERATIONS = 2;

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

    DLBABP_enqueue(kernel_ctx, dlbabp, &t, maxl);
    /* ---------- end of initOctree ---------- */    
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (local_id == 0) {
    localStealAttempts = 0;
  }
  
  /* /\* Hugues: do the octree partitionning several times to last longer *\/ */
  /* while (NUM_ITERATIONS > 0) { */

  /* if (k_get_global_id(kernel_ctx) == 0) { */
  /*   //NUM_ITERATIONS--; */
  /*   octree_init(kernel_ctx, dlbabp, treeSize, particlesDone, maxl, stealAttempts, num_pools, numParticles); */
  /* } */
  //barrier(CLK_GLOBAL_MEM_FENCE);
  
  /* main loop */
  while (true) {
    barrier(CLK_LOCAL_MEM_FENCE);

    /* can be killed if pool is empty */
    int group_id = k_get_global_id(kernel_ctx);
    if (emptyPool(dlbabp, group_id)) {
      /* FIXME Hugues: here use __ckill() rather than ckill() since
         ckill() macro terminates with 'return -1', which is invalid
         here. I do not change the ckill() macro since this return value
         is currently used in the implementation of global_barrier_*(),
         see iw_barrier.cl source file */
      if (__ckill(kernel_ctx, scheduler_ctx, scratchpad, group_id) == -1) {
        return;
      }
    }

    // Try to acquire new task
    if (DLBABP_dequeue(kernel_ctx, dlbabp, &t, randdata, &localStealAttempts, num_pools) == 0) {
      check = *particlesDone;
      barrier(CLK_LOCAL_MEM_FENCE);
      if (check == numParticles) {
        if (local_id == 0) {
          atomic_add(stealAttempts, localStealAttempts);
        }
        return;
      }
      continue;
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
          DLBABP_enqueue(kernel_ctx, dlbabp, &newTask, maxl);
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
  }
    //} // while NUM_ITERATIONS
}

/* ========================================================================= */

__kernel void mega_kernel(
                          // Graphics kernel args
                          int graphics_length,
                          __global int * graphics_buffer,
                          __global atomic_int * graphics_result,
						  
                          // Persistent kernel args
                          __global DLBABP* dlbabp,
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
#define PERSISTENT_KERNEL octree_main(persistent_kernel_ctx, s_ctx, scratchpad, dlbabp, randdata, maxl, particles, newparticles, tree, numParticles, treeSize, particlesDone, maxchilds, stealAttempts, num_pools, deq, dh, maxlength);

  // Everything else is in here	
#include "main_device_body.cl"
}
//
