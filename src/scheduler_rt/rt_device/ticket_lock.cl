#pragma once

#include "../rt_common/ticket_lock.h"

void TL_lock(Ticket_lock *m) {
  //get our ticket
  int ticket = atomic_fetch_add_explicit(&(m->counter), 1, memory_order_acq_rel, memory_scope_device);

  //spin while our ticket isn't now_serving
  while (atomic_load_explicit(&(m->now_serving), memory_order_relaxed, memory_scope_device) != ticket);

  //synchronise 
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
}


void TL_unlock(Ticket_lock *m) {
  //get the now serving value
  int tmp = atomic_load_explicit(&(m->now_serving), memory_order_relaxed, memory_scope_device);

  //increment it
  tmp +=1;

  //release the new value
  atomic_store_explicit(&(m->now_serving),tmp, memory_order_release, memory_scope_device);
}