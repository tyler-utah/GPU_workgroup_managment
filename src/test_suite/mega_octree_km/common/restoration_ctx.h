#pragma once
typedef struct {
  CL_UCHAR_TYPE target;
  unsigned int local_id;
  unsigned int local_size;
  unsigned int localStealAttempts;
  CL_INT_TYPE i;
} Restoration_ctx;

