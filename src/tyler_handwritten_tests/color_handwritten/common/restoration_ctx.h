#pragma once

/*typedef struct {
  CL_UCHAR_TYPE target;
  CL_INT_TYPE graph_color;
  CL_INT_TYPE i;
  MY_CL_GLOBAL CL_INT_TYPE * read_stop;
  MY_CL_GLOBAL CL_INT_TYPE * write_stop;
  MY_CL_GLOBAL CL_INT_TYPE * swap;
} Restoration_ctx;
*/

typedef struct {
  CL_UCHAR_TYPE target;
  MY_CL_GLOBAL CL_INT_TYPE * write_stop;
  MY_CL_GLOBAL CL_INT_TYPE * read_stop;
  MY_CL_GLOBAL CL_INT_TYPE * swap;
  CL_INT_TYPE graph_color;
} Restoration_ctx;

