#pragma once

#define FULL_FENCE (CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#define BARRIER barrier(FULL_FENCE)

#ifdef AMD_MEM_ORDERS
#define memory_order_special_relax_acquire memory_order_acquire
#else
#define memory_order_special_relax_acquire memory_order_relaxed
#endif