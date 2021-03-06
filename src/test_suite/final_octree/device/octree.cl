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

int pool_try_lock(__global atomic_int *task_pool_lock, int pool_id)
{
  int expected = false;
  return atomic_compare_exchange_strong(&(task_pool_lock[pool_id]), &expected, true);
}

/*---------------------------------------------------------------------------*/

void pool_unlock(__global atomic_int *task_pool_lock, int pool_id)
{
  atomic_store(&(task_pool_lock[pool_id]), false);
}

/*---------------------------------------------------------------------------*/

/* wgm_task_pop() MUST be called by local_id 0. This function grabs a
   task from a pool and stores it in the task argument. Returns true if
   it works.  must local_fence after this function */
int wgm_task_pop(__local Task *task, __global Task *pools, __global atomic_int *task_pool_lock, __global int *pool_head, const int pool_size, int pool_id)
{
  int poped = false;
  /* spinwait on the pool lock */
  while (!(pool_try_lock(task_pool_lock, pool_id)));
  /* If pool is not empty, pick up the latest inserted task. */
  if (pool_head[pool_id] > 0) {
    pool_head[pool_id] -= 1;
    *task = pools[(pool_size * pool_id) +  pool_head[pool_id]];
    poped = true;
  }
  pool_unlock(task_pool_lock, pool_id);
  return poped;
}

/*---------------------------------------------------------------------------*/

/* wgm_task_push() adds the task argument to the indicated pool. Returns
   true if it worked. */
int wgm_task_push(__local Task *task, __global Task *pools, __global atomic_int *task_pool_lock, __global int *pool_head, const int pool_size, int pool_id)
{
  int pushed = false;
  /* spinwait on the pool lock */
  while (!(pool_try_lock(task_pool_lock, pool_id)));
  /* If pool is not full, insert task */
  if (pool_head[pool_id] < pool_size) {
    pools[(pool_size * pool_id) +  pool_head[pool_id]] = *task;
    pool_head[pool_id] += 1;
    pushed = true;
  }
  pool_unlock(task_pool_lock, pool_id);
  return pushed;
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
                 __global Task *pools,
                 __global atomic_int *task_pool_lock,
                 __global int *pool_head,
                 const int num_pools,
                 const int pool_size,
                 __global atomic_uint* treeSize,
                 __global atomic_uint* particlesDone,
                 unsigned int numParticles,
                 __local Task *t
                 )
{
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
  t->treepos=0;
  t->middle.x=0;
  t->middle.y=0;
  t->middle.z=0;
  t->middle.w=256;

  t->beg = 0;
  t->end = numParticles;
  t->flip = false;

  /* push in pool_id zero */
  wgm_task_push(t, pools, task_pool_lock, pool_head, pool_size, 0);
  /* ---------- end of initOctree ---------- */
}

/*---------------------------------------------------------------------------*/

__kernel void octree_main (
                  /* octree args */
                  __global float4* particles,
                  __global float4* newparticles,
                  __global unsigned int* tree,
                  const uint numParticles,
                  __global atomic_uint* treeSize,
                  __global atomic_uint* particlesDone,
                  const unsigned int maxchilds,
                  __global Task *pools,
                  __global atomic_int *task_pool_lock,
                  __global int *pool_head,
                  const int num_pools,
                  const int pool_size,
                  __global float4* frompart,
                  __global float4* topart
                  )
{
  __local uint count[8];
  __local uint sum[8];
  __local Task t[1];
  __local int got_new_task[1];
  __local Task newTask[1];
  __local int game_over[1];

  /* ADD INIT HERE */
  if (get_global_id(0) == 0) {
    octree_init(pools, task_pool_lock, pool_head, num_pools, pool_size, treeSize, particlesDone, numParticles, &(t[0]));
  }
  global_barrier();

  /* main loop */
  while (true) {

    /* only the first group can fork, to limit calls to offer_fork. */
    if (get_group_id(0) == 0) {
      offer_fork();
    }

    // can be killed before handling a task, but always keep at least
    // one work-group alive.
    if (get_group_id(0) > 0) {
      offer_kill();
    }

    uint local_id = get_local_id(0);
    uint local_size = get_local_size(0);
    int pool_id = get_group_id(0) % num_pools;

    // Try to acquire new task
    if (local_id == 0) {
      // First try own task, then check neighbours
      for (int i = 0; i < num_pools; i++) {
        got_new_task[0] = wgm_task_pop(&(t[0]), pools, task_pool_lock, pool_head, pool_size, (pool_id + i) % num_pools);
        if (got_new_task[0]) {
          break;
        }
      }
      game_over[0] = false;
      if (!got_new_task[0]) {
        /* test for end of computation */
        game_over[0] = atomic_load(particlesDone) >= numParticles;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (!got_new_task[0]) {
      if (game_over[0]) {
        break;
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

    for(int i = (t[0]).beg + local_id; i < (t[0]).end; i += local_size) {
      int box = whichbox(frompart[i],(t[0]).middle);
      atomic_add(&(count[box]), 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
      sum[0] = count[0];
      for (int x = 1; x < 8; x++) {
        sum[x] = sum[x-1] + count[x];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = (t[0]).beg + local_id; i < (t[0]).end; i += local_size) {
      int toidx = (t[0]).beg + atomic_dec(&(sum[whichbox(frompart[i],(t[0]).middle)])) - 1;
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
            pushed = wgm_task_push(&(newTask[0]), pools, task_pool_lock, pool_head, pool_size, (pool_id + j) % num_pools);
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
          for (
               int j = (t[0]).beg + sum[i] + local_id;
               j < (t[0]).beg + sum[i] + count[i];
               j += local_size)
            {
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

  } // end of main loop
}

/*---------------------------------------------------------------------------*/
