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

/**
 * scanning strategy: scan in 4 directions.
 *  - right: one thread per row => 6 threads
 *  - down: one thread per col => 7 threads
 *  - down_right: one thread per diagonal that may contains 4
 *                tokens => 6 threads
 *  - down_left: one thread per diagonal that may contains 4
 *               tokens => 6 threads
 */
const int max_scan_id = 6+7+6+6;

/*---------------------------------------------------------------------------*/

typedef enum {
  RIGHT,
  DOWN,
  DOWN_RIGHT,
  DOWN_LEFT,
} Direction;

/*---------------------------------------------------------------------------*/

/* Hugues: different ponderation for chain of length 2 and 3 ? */
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

int scan_board(__local uchar *board, int row, int col, Direction direction) {
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

/* Only wgmaster return value is meaningful */
int board_value(__local uchar *board, __local int *val, int local_id, int local_size)
{
  for (int i = local_id; i < max_scan_id; i += local_size) {
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
    if ((6+7+6) <= i && i < max_scan_id) {
      /* scan diagonal left */
      i -= (6+7+6);
      if (i < 3) {
        tmpval = scan_board(board, i, (NUM_COL - 1), DOWN_LEFT);
      } else {
        /* Warning: here i is ((i - 3) + 3), draw a board to see */
        tmpval = scan_board(board, 0, i, DOWN_LEFT);
      }
    }

    /* update val */
    if (tmpval == PLUS_INF || tmpval == MINUS_INF) {
      /* found a winner */
      val[local_id] = tmpval;
      break;
    } else {
      val[local_id] += tmpval;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  /* wgmaster reduces */
  if (local_id == 0) {
    int res = 0;
    for (int i = 0; i < local_size; i++) {
      if (val[i] == PLUS_INF || val[i] == MINUS_INF) {
        return val[i];
      }
      res += val[i];
    }
    return res;
  }

  /* only wgmaster is meaningful */
  return 0;
}

/*---------------------------------------------------------------------------*/

void play_moves(__local uchar *board, uchar *move, int num_move)
{
  /* we explore possible moves for the computer, so computer is always
     the first to play */
  uchar player = COMPUTER;

  for (int i = 0; i < num_move; i++) {
    int col = move[i];
    int row = 0;

    /* special case: columns is full, ignore invalid move */
    if (board[col] != EMPTY) {
      continue;
    }

    while (row < NUM_ROW &&
           board[(row * NUM_COL) + col] == EMPTY) {
      row++;
    }
    /* place token one row above */
    board[((row-1) * NUM_COL) + col] = player;

    /* update player */
    if (player == COMPUTER) {
      player = HUMAN;
    } else {
      player = COMPUTER;
    }
  }
}

/*---------------------------------------------------------------------------*/

__kernel void
connect_four(
             __global uchar *base_board,
             __global int *values
             )
{
  __local int val[256];
  __local uchar board[NUM_CELL];
  /* Moves are stored in uchar since possible values are in 0..6 */
  uchar moves[8];

  int local_id = get_local_id(0);
  int local_size = get_local_size(0);
  int group_id = get_group_id(0);

  if (local_id == 0) {
    for (int i = 0; i < NUM_CELL; i++) {
      board[i] = base_board[i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (group_id < NUM_COL) {

    /* wgmaster rebuild the local board */
    if (local_id == 0) {
      moves[0] = group_id;
      play_moves(board, moves, 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* get value */
    int value = board_value(board, val, local_id, local_size);
    if (local_id == 0) {
      values[group_id] = value;
    }
  }

  /* barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); */

  /* if (get_global_id(0) == 0) { */
  /*   for (int i = 0; i < NUM_CELL; i++) { */
  /*     base_board[i] = board[i]; */
  /*   } */
  /* } */

  /* barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); */
}

/*---------------------------------------------------------------------------*/
