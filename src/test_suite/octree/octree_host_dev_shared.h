/* type declaration shared between host and device */

/*
  For host code, requires:
  #include <CL/cl.hpp>
*/

/*---------------------------------------------------------------------------*/
/* Task */

typedef struct {

#ifdef __OPENCL_C_VERSION__
  float4 middle;
#else
  cl_float4 middle;
#endif

  bool flip;
  unsigned int end;
  unsigned int beg;
  unsigned int treepos;
} Task;

/*---------------------------------------------------------------------------*/
/* class DLBABP */

typedef struct {
  volatile int tail;
  volatile int head;
} DequeHeader;

/*---------------------------------------------------------------------------*/
/* DLBABP */

typedef struct {
#ifdef __OPENCL_C_VERSION__
  /* Hugues: pointers to other buffers declared at the host side,
   * therefore __global. Moreover, for the 'dh' variable, __global is
   * required for atomic_cmpxchg() later on */
  __global Task *deq;
  __global DequeHeader* dh;
#else
  Task *deq;
  DequeHeader* dh;
#endif
  unsigned int maxlength;
} DLBABP;

/*---------------------------------------------------------------------------*/
