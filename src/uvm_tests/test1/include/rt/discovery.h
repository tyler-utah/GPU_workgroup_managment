#pragma once

#include "../rt_common/discovery.h"


void mk_init_discovery_ctx(Discovery_ctx *init) {
  init->count = 0;
  init->poll_open = 1;
  init->m.counter = 0;
  init->m.now_serving = 0;
  for (int i = 0; i < MAX_P_GROUPS; i++) {
	  init->p_group_ids[i] = -1;
  }
}