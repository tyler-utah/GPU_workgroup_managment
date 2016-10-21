#pragma once

/*
  FIXME: list OpenCL types exhaustively, see OpenCL C spec tables 6.1,
  6.2 and section 6 in general
*/

#ifdef __OPENCL_C_VERSION__

/* OpenCL version */

#define MY_CL_GLOBAL __global

#define CL_CHAR_TYPE char
#define CL_UCHAR_TYPE unsigned char
#define CL_SHORT_TYPE short
#define CL_USHORT_TYPE unsigned short
#define CL_INT_TYPE int
#define CL_UINT_TYPE unsigned int
#define CL_LONG_TYPE long
#define CL_ULONG_TYPE unsigned long
#define CL_FLOAT_TYPE float
#define CL_DOUBLE_TYPE double
#define CL_HALF_TYPE half

#define ATOMIC_CL_INT_TYPE atomic_int

#else

/* C++ version */

#define MY_CL_GLOBAL

#define CL_CHAR_TYPE cl_char
#define CL_UCHAR_TYPE cl_uchar
#define CL_SHORT_TYPE cl_short
#define CL_USHORT_TYPE cl_ushort
#define CL_INT_TYPE cl_int
#define CL_UINT_TYPE cl_uint
#define CL_LONG_TYPE cl_long
#define CL_ULONG_TYPE cl_ulong
#define CL_FLOAT_TYPE cl_float
#define CL_DOUBLE_TYPE cl_double
#define CL_HALF_TYPE cl_half

#define ATOMIC_CL_INT_TYPE cl_int

#endif
