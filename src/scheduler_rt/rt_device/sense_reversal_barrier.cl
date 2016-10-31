#pragma once

// Sense reversal barrier: inter-workgroup barier, does not resize the
// number of workgroups, but is robust to fork/kill of workgroups
// between barrier calls.
void global_barrier_sense_reversal(__global sense_reversal_barrier *bar, __local int *sense, __global Kernel_ctx *k_ctx)
{
  if (get_local_id(0) == 0) {
    /* flip the sense */
    *sense = !(*sense);
    if (atomic_fetch_add(&(bar->counter), 1) == 0) {
      /* only the first to hit the barrier enters here. it spins waiting
         for the other workgroups to arrive. The number of workgroups is
         dynamic, so it should be checked from kernel_ctx everytime */
      while (true) {
        /* Here we MUST first load the barrier counter. If not, the
           following can happen: load the number of groups, say it's
           equal to n. Then, concurrently, the scheduler allocates a new
           group, so now the number of groups is (n+1), and n groups
           enter the barrier. Now the first workgroup that has hitted
           the barrier resumes and load the barrier counter, which is
           equal to n. Therefore, it releases everybody, although it
           should have waited for (n+1) groups. */
        int bar_counter = atomic_load(&(bar->counter));
        int num_groups = k_get_num_groups(k_ctx);
        if (bar_counter == num_groups) {
          /* everyone is here, first reset the counter */
          atomic_store(&(bar->counter), 0);
          /* then assign the sense to release others */
          atomic_store(&(bar->sense), *sense);
          break;
        }
      }
    } else {
      /* spin on the sense flag */
      while (*sense != atomic_load(&(bar->sense)));
    }
  }

  /* Synchronize threads within a group. */
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
