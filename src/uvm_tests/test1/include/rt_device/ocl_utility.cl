#pragma once

#define FULL_FENCE (CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#define BARRIER barrier(FULL_FENCE)
