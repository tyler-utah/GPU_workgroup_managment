#include "../rt_common/cl_types.h"
#include "connect_four_data.h"

/**
   The board is indexed this way:

     0 1 2 3 4 5 6
   0 . . . . . . .
   1 . . . . . . .
   2 . . . . . . .
   3 . . . . . . .
   4 . . . . . . .
   5 . . . . . . .

*/

/*---------------------------------------------------------------------------*/

typedef enum {
  RIGHT,
  DOWN,
  DOWN_RIGHT,
  DOWN_LEFT,
} Direction;

/*---------------------------------------------------------------------------*/

void chain_value(int *value, uchar player, int chain) {
  if (player != EMPTY) {
    if (4 <= chain) {
      *value = (player == COMPUTER) ? PLUS_INF : MINUS_INF;
    } else if (2 <= chain) {
      *value += (player == COMPUTER) ? 1 : -1;
    }
  }
}

/*---------------------------------------------------------------------------*/

int scan_board(__global uchar *board, int row, int col, Direction direction) {
  int res = 0;
  uchar player = EMPTY;
  int chain = 0;
  do {
    int cell = board[(row * NUM_COL) + col];
    if (cell == player) {
      chain++;
    } else {
      /* chain is broken by the cell, check its value */
      chain_value(&res, player, chain);
      if (res == PLUS_INF || res == MINUS_INF) {
        /* A winning config was detected */
        return res;
      }
      /* init for a new chain */
      player = cell;
      chain = 1;
    }

    /* advance position */
    switch (direction) {
    case RIGHT:
      col++;
      break;
    case DOWN:
      row++;
      break;
    case DOWN_RIGHT:
      row++;
      col++;
      break;
    case DOWN_LEFT:
      row++;
      col--;
      break;
    }
  } while (0 <= row && row < NUM_ROW &&
           0 <= col && col < NUM_COL);

  /* We reached the board end, check the chain again */
  chain_value(&res, player, chain);
  return res;
}

/*---------------------------------------------------------------------------*/

void board_value(__global uchar *board, __global int *value, __local int *val)
{
  /* a task is operated within one workgroup */
  if (get_group_id(0) == 0) {
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    /**
     * scanning strategy: scan in 4 directions.
     *  - right: one thread per row => 6 threads
     *  - down: one thread per col => 7 threads
     *  - down_right: one thread per diagonal that may contains 4
     *                tokens => 6 threads
     *  - down_left: one thread per diagonal that may contains 4
     *               tokens => 6 threads
     */

    int maxid = 6+7+6+6;
    int i;

    val[lid] = 0;

    for (i = lid; i < maxid; i += local_size) {
      int tmpval;

      if (0 <= i && i < 6) {
        /* scan row */
        tmpval = scan_board(board, i, 0, RIGHT);
      }
      if (6 <= i && i < (6+7)) {
        /* scan col */
        i -= 6;
        tmpval = scan_board(board, 0, i, DOWN);
      }
      if ((6+7) <= i && i < (6+7+6)) {
        /* scan diagonal right */
        i -= (6+7);
        if (i < 3) {
          tmpval = scan_board(board, i, 0, DOWN_RIGHT);
        } else {
          tmpval = scan_board(board, 0, (i - 2), DOWN_RIGHT);
        }
      }
      if ((6+7+6) <= i && i < maxid) {
        /* scan diagonal left */
        i -= (6+7+6);
        if (i < 3) {
          tmpval = scan_board(board, i, (NUM_COL - 1), DOWN_LEFT);
        } else {
          /* Warning: here i is ((i - 3) + 3), draw a board to see */
          tmpval = scan_board(board, 0, i, DOWN_LEFT);
        }
      }

      /* update value */
      if (tmpval == PLUS_INF || tmpval == MINUS_INF) {
        /* found a winner */
        val[lid] = tmpval;
        break;
      } else {
        val[lid] += tmpval;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (lid == 0) {
      for (i = 0; i < maxid; i++) {
        *value += val[i];
        if (val[i] == PLUS_INF || val[i] == MINUS_INF) {
          *value = val[i];
          break;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

__kernel void
connect_four(
             __global uchar *board,
             __global int *value
             )
{
  __local int val[256];
  board_value(board, value, val);
}

/*---------------------------------------------------------------------------*/
