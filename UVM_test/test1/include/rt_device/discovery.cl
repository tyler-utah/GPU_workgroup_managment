#pragma once

#include "../rt_common/discovery.h"
#include "ocl_utility.cl"
#include "ticket_lock.cl"

#define PARTICIPATING_FLAG 1
#define NON_PARTICIPATING_FLAG 0

int discovery_protocol_rep(__global Discovery_ctx *d_ctx) {

  //polling phase
  int id;
  TL_lock(&(d_ctx->m));
  if ((d_ctx->poll_open)) {
	id = (d_ctx->count);
    (d_ctx->count) = id + 1;
    TL_unlock(&(d_ctx->m));
	d_ctx->p_group_ids[get_group_id(0)] = id;
  }
  else {
    TL_unlock(&(d_ctx->m));
    return NON_PARTICIPATING_FLAG;
  }
  
  //Closing phase
  TL_lock(&(d_ctx->m));
  
    if ((d_ctx->poll_open)) {
      d_ctx->poll_open = 0;
  }
  TL_unlock(&(d_ctx->m));

  return PARTICIPATING_FLAG;
}

int discovery_protocol(__global Discovery_ctx *d_ctx) {
  __local int ret_flag;
  int id = get_local_id(0);
  if (id == 0) {
    ret_flag = discovery_protocol_rep(d_ctx);
  }
  BARRIER;
  return ret_flag;
}

#define DISCOVERY_PROTOCOL(d_ctx)                             \
	if (discovery_protocol(d_ctx) == NON_PARTICIPATING_FLAG)  \
	    return;                                               