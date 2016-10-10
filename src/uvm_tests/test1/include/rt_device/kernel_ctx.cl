#pragma once

#include "../rt_common/kernel_ctx.h"

// For now this is just atomic. Don't know how this affects the graphics kernel.
int k_get_num_groups(__global Kernel_ctx * k_ctx) {
	return atomic_load_explicit(&(k_ctx->num_groups), memory_order_relaxed, memory_scope_device);
}

int k_get_num_groups_acquire(__global Kernel_ctx * k_ctx) {
	return atomic_load_explicit(&(k_ctx->num_groups), memory_order_acquire, memory_scope_device);
}

int k_get_global_size(__global Kernel_ctx * k_ctx) {
	return k_get_num_groups(k_ctx) * get_local_size(0); 
}

int k_get_group_id(__global Kernel_ctx * k_ctx) {
	return k_ctx->group_ids[p_get_group_id(k_ctx->d_ctx)];
}

int k_get_global_id(__global Kernel_ctx * k_ctx) {
	return k_get_group_id(k_ctx) * get_local_size(0) + get_local_id(0);
}