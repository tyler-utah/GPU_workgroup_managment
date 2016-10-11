#include "octree_host_dev_shared.h"

/*===========================================================================*/
/* rand */

int myrand(__global int *randdata) {
  int id = get_group_id(0);
  randdata[id] = randdata[id] * 1103515245 + 12345;
  return((unsigned)(randdata[id] / 65536) % 32768) + id;
}

/*===========================================================================*/
/* lbabp */

void DLBABP_push(__global DLBABP *dlbabp, __local Task *val, __global volatile int *maxl) {
  int id = get_group_id(0);
  dlbabp->deq[id * dlbabp->maxlength + dlbabp->dh[id].tail] = *val;
  dlbabp->dh[id].tail++;

  if (*maxl < dlbabp->dh[id].tail) {
    atomic_max(maxl, dlbabp->dh[id].tail);
  }
}

/*---------------------------------------------------------------------------*/

void DLBABP_enqueue(__global DLBABP *dlbabp, __local Task *val, __global volatile int *maxl) {
  /* Hugues todo: check calls to DLBABP_enqueue, can any other thread
   * than id0 can call it ? */
  if (get_local_id(0) == 0) {
    DLBABP_push(dlbabp, val, maxl);
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
  int localTail;
  int oldHead;
  int newHead;

  oldHead = dlbabp->dh[idx].head;
  localTail = dlbabp->dh[idx].tail;
  if(localTail <= getIndex(oldHead)) {
    return -1;
  }

  *val = dlbabp->deq[idx * dlbabp->maxlength + getIndex(oldHead)];
  newHead = incIndex(oldHead);
  if (atomic_cmpxchg(&(dlbabp->dh[idx].head), oldHead, newHead) == oldHead) {
    return 1;
  }

  return -1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_pop(__global DLBABP *dlbabp, __local Task *val) {
  int localTail;
  int oldHead;
  int newHead;
  int id = get_group_id(0);

  localTail = dlbabp->dh[id].tail;
  if(localTail == 0) {
    return -1;
  }

  localTail--;

  dlbabp->dh[id].tail = localTail;

  *val = dlbabp->deq[id * dlbabp->maxlength + localTail];

  oldHead = dlbabp->dh[id].head;

  if (localTail > getIndex(oldHead)) {
    return 1;
  }

  dlbabp->dh[id].tail = 0;
  newHead = getZeroIndexIncCtr(oldHead);
  if(localTail == getIndex(oldHead)) {
    if(atomic_cmpxchg(&(dlbabp->dh[id].head), oldHead, newHead) == oldHead) {
      return 1;
    }
  }
  dlbabp->dh[id].head = newHead;
  return -1;
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue2(__global DLBABP *dlbabp, __local Task *val, __global int *randdata, unsigned int *localStealAttempts)
{
  if (DLBABP_pop(dlbabp, val) == 1) {
    return 1;
  }

  *localStealAttempts += 1;

  if (DLBABP_steal(dlbabp, val, myrand(randdata)%get_num_groups(0)) == 1) {
    return 1;
  } else {
    return 0;
  }
}

/*---------------------------------------------------------------------------*/

int DLBABP_dequeue(__global DLBABP *dlbabp, __local Task *val, __global int *randdata, unsigned int *localStealAttempts) {
  __local volatile int rval;
  int dval = 0;

  if(get_local_id(0) == 0) {
    rval = DLBABP_dequeue2(dlbabp, val, randdata, localStealAttempts);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  dval = rval;
  barrier(CLK_LOCAL_MEM_FENCE);

  return dval;
}

/*===========================================================================*/
/* Octree Kernel */

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

__kernel void makeOctree(
  __global DLBABP* dlbabp,
  __global int *randdata,
  __global volatile int *maxl,
  __global float4* particles,
  __global float4* newparticles,
  __global unsigned int* tree,
  unsigned int particleCount,
  __global unsigned int* treeSize,
  __global unsigned int* particlesDone,
  unsigned int maxchilds,
  int isStatic,
  __global unsigned int *stealAttempts)
{
  /* Hugues todo: in OpenCL, isStatic is always false since we only
   * implement dynamic version (i.e., work-stealing) */

  /* Hugues: in Cuda version, frompart and topart are __local (i.e.,
   * __shared__), but here the OpenCL compiler complains if I declare
   * them as __local since we assign particles and newparticles to it */
  __global float4* frompart;
  __global float4* topart;

  __local unsigned int count[8];
  __local int sum[8];

  __local Task t;
  __local unsigned int check;

  unsigned int local_id = get_local_id(0);
  unsigned int local_size = get_local_size(0);

  unsigned int localStealAttempts;
  if (local_id == 0) {
    localStealAttempts = 0;
  }

  while (true) {
    barrier(CLK_LOCAL_MEM_FENCE);

    // Try to acquire new task
    if (DLBABP_dequeue(dlbabp, &t, randdata, &localStealAttempts) == 0) {
      if (isStatic)
        return;
      check = *particlesDone;
      barrier(CLK_LOCAL_MEM_FENCE);
      if (check == particleCount) {
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

    /* Hugues: here we use global since we've just updated frompart /
     * topart */
    barrier(CLK_GLOBAL_MEM_FENCE);

    for(int i = local_id; i < 8; i += local_size) {
      count[i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = t.beg + local_id; i < t.end; i += local_size) {
      atomic_inc(&count[whichbox(frompart[i],t.middle)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
      sum[0] = count[0];
      for (int x = 1; x < 8; x++)
        sum[x] = sum[x-1] + count[x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = t.beg + local_id; i < t.end; i += local_size) {
      int toidx = t.beg + atomic_dec(&sum[whichbox(frompart[i],t.middle)]) - 1;
      topart[toidx] = frompart[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < 8; i++) {
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
          DLBABP_enqueue(dlbabp, &newTask, maxl);
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
}

/*---------------------------------------------------------------------------*/

__kernel void initDLBABP(
  __global DLBABP *dlbabp,
  __global Task *deq,
  __global DequeHeader *dh,
  unsigned int maxlength)
{
  /* Hugues: only one thread might touch global memory */
  if (get_global_id(0) == 0) {
    dlbabp->deq = deq;
    dlbabp->dh = dh;
    dlbabp->maxlength = maxlength;
  }
}

/*---------------------------------------------------------------------------*/

__kernel void initOctree(
  __global DLBABP *dlbabp,
  __global volatile int *maxl,
  __global unsigned int *stealAttempts,
  __global unsigned int *treeSize,
  __global unsigned int* particlesDone,
  int numParticles)
{
  *treeSize = 100;
  *particlesDone = 0;
  /* In Cuda, maxl is a kernel global initialized to 0 */
  *maxl = 0;
  *stealAttempts = 0;

  __local Task t;

  t.treepos=0;
  t.middle.x=0;
  t.middle.y=0;
  t.middle.z=0;
  t.middle.w=256;

  t.beg = 0;
  t.end = numParticles;
  t.flip = false;

  DLBABP_enqueue(dlbabp, &t, maxl);
}

/*---------------------------------------------------------------------------*/
