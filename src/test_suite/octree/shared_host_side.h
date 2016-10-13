/* requires:
   #include <CL/cl.hpp>
*/

/*
 * Host types must be kept in sync with device counterparts, which are
 * declared in the kernel file
 */

/*---------------------------------------------------------------------------*/
/* Task */

typedef struct {
  cl_float4 middle;
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
  Task *deq;
  DequeHeader* dh;
  unsigned int maxlength;
} DLBABP;

/*---------------------------------------------------------------------------*/
