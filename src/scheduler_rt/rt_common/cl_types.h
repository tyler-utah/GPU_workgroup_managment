#pragma once

#ifdef __OPENCL_C_VERSION__

/* OpenCL version */

#define CL_CHAR_TYPE char
#define CL_UCHAR_TYPE unsigned char
#define CL_SHORT_TYPE short
#define CL_USHORT_TYPE unsigned short
#define CL_INT_TYPE int
#define CL_UINT_TYPE unsigned int
#define CL_LONG_TYPE long
#define CL_ULONG_TYPE unsigned long

#define MY_CL_GLOBAL __global

#else

/* C++ version */

#define CL_CHAR_TYPE cl_char
#define CL_UCHAR_TYPE cl_uchar
#define CL_SHORT_TYPE cl_short
#define CL_USHORT_TYPE cl_ushort
#define CL_INT_TYPE cl_int
#define CL_UINT_TYPE cl_uint
#define CL_LONG_TYPE cl_long
#define CL_ULONG_TYPE cl_ulong

#define MY_GLOBAL

#endif
