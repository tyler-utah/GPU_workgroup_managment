
/*---------------------------------------------------------------------------*/

typedef struct {
  CL_FLOAT4_TYPE middle;
  CL_BOOL_TYPE flip;
  CL_UINT_TYPE end;
  CL_UINT_TYPE beg;
  CL_UINT_TYPE treepos;
} Task;

/*---------------------------------------------------------------------------*/

typedef struct {
  ATOMIC_CL_INT_TYPE tail;
  ATOMIC_CL_INT_TYPE head;
} DequeHeader;

/*---------------------------------------------------------------------------*/
