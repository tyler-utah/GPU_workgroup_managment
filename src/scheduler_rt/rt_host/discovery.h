#pragma once

#include "../rt_common/discovery.h"


void mk_init_discovery_ctx(Discovery_ctx *init) {
	//std::cout << "6666" << std::endl;
  init->count = 0;
  init->poll_open = 1;
  init->m.counter = 0;
  init->m.now_serving = 0;
  init->im_exit = 0;
  for (int i = 0; i < MAX_P_GROUPS; i++) {
	  init->p_group_ids[i] = -1;
  }
}

void mk_init_discovery_ctx_occupancy(Discovery_ctx *init) {
	//std::cout << "5555" << std::endl;

	init->count = 0;
	init->poll_open = 1;
	init->m.counter = 0;
	init->m.now_serving = 0;
	init->im_exit = 1;
	for (int i = 0; i < MAX_P_GROUPS; i++) {
		init->p_group_ids[i] = -1;
	}
}