/*---------------------------------------------------------------------------*/

#define EMPTY ((CL_UCHAR_TYPE)0)
#define HUMAN ((CL_UCHAR_TYPE)1)
#define COMPUTER ((CL_UCHAR_TYPE)2)

/*---------------------------------------------------------------------------*/

/* Warning: there is still some parts of code where these values are
   hardcoded */
const CL_INT_TYPE NUM_ROW = 6;
const CL_INT_TYPE NUM_COL = 7;
const CL_INT_TYPE NUM_CELL = 6 * 7;

const CL_INT_TYPE PLUS_INF = 666;
const CL_INT_TYPE MINUS_INF = -666;

const CL_INT_TYPE MAX_LOOKAHEAD = 3;

/*---------------------------------------------------------------------------*/

typedef struct {
  CL_INT_TYPE parent;
  CL_INT_TYPE level;
  CL_UCHAR_TYPE moves[MAX_LOOKAHEAD];
  ATOMIC_CL_INT_TYPE value;
  ATOMIC_CL_INT_TYPE num_child_answer;
} Node;

/*---------------------------------------------------------------------------*/
