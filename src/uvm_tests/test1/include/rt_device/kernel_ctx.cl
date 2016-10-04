#pragma once

#include "../rt_common/kernel_ctx.h"

int k_get_num_groups(__global Kernel_ctx * k_ctx) {
	return k_ctx->num_groups;
}

int k_get_global_size(__global Kernel_ctx * k_ctx) {
	return k_ctx->num_groups * get_local_size(0); 
}

int k_get_group_id(__global Kernel_ctx * k_ctx) {
	return k_ctx->group_ids[p_get_group_id(k_ctx->d_ctx)];
}

int k_get_group_id_disc(__global Kernel_ctx * k_ctx, __global Discovery_ctx *d_ctx) {
	return k_ctx->group_ids[p_get_group_id(d_ctx)];
}


int k_get_global_id(__global Kernel_ctx * k_ctx) {
	return k_get_group_id(k_ctx) * get_local_size(0) + get_local_id(0);
}