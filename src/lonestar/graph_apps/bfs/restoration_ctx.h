#pragma once
typedef struct {
  CL_UCHAR_TYPE target;
  CL_INT_TYPE iteration;
  MY_CL_GLOBAL CL_INT_TYPE * in_wl;
  MY_CL_GLOBAL CL_INT_TYPE * out_wl;
  MY_CL_GLOBAL CL_INT_TYPE * in_index;
  MY_CL_GLOBAL CL_INT_TYPE * out_index;
} Restoration_ctx;

