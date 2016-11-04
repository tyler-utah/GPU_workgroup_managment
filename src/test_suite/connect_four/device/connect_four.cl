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
  val[local_id] = 0;

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

  /* only wgmaster return value is meaningful */
  return 0;
}

/*---------------------------------------------------------------------------*/

/* update_board() replay moves of node from the base board and store the
   resulting board in the local board */
void update_board(__local uchar *board, __global uchar *base_board, __global Node *node, int local_id, int local_size)
{
  /* start from base board */
  for (int i = local_id; i < NUM_CELL; i += local_size) {
    board[i] = base_board[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* moves must be played sequentially, so wgmaster thread does it */
  if (local_id == 0) {

    /* we explore possible moves for the computer, so computer is always
       the first to play */
    uchar player = COMPUTER;

    /* play moves */
    for (int i = 0; i < node->level; i++) {
      int col = node->moves[i];
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
  barrier(CLK_LOCAL_MEM_FENCE);
}

/*---------------------------------------------------------------------------*/

/* compute_node_value() returns the value of the board given the node
   moves, this value is also stored in the corresponding node
   field. This function does not compute values of potential children
   nodes. */
int compute_node_value(__global uchar *base_board, __local uchar *board, __local int *val, __global Node *node, int local_id, int local_size)
{
  update_board(board, base_board, node, local_id, local_size);
  int value = board_value(board, val, local_id, local_size);
  if (local_id == 0) {
    atomic_store(&(node->value), value);
    /* broadcast value to all nodes through val */
    val[0] = value;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  value = val[0];
  return value;
}

/*---------------------------------------------------------------------------*/

void create_children(__local int *child_id, __global Node *nodes, __global atomic_int *node_head, int parent_id, int local_id, int local_size)
{
  __global Node *parent = &(nodes[parent_id]);
  for (int i = local_id; i < NUM_COL; i += local_size) {
    int n = atomic_fetch_add(node_head, 1);
    /* safety */
    if (n >= NUM_NODE) {
      break;
    }
    Node *child = &(nodes[n]);
    child->parent = parent_id;
    child->level = parent->level + 1;
    for (int j = 0; j < parent->level; j++) {
      child->moves[j] = parent->moves[j];
    }
    child->moves[parent->level] = i;
    atomic_store(&(child->value), 0);
    atomic_store(&(child->num_child_answer), 0);
    child_id[i] = n;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

/*---------------------------------------------------------------------------*/

bool try_lock(__global atomic_int *task_pool_lock, int pool_id)
{
  int expected = false;
  return atomic_compare_exchange_strong(&(task_pool_lock[pool_id]), &expected, true);
}

/*---------------------------------------------------------------------------*/

void unlock(__global atomic_int *task_pool_lock, int pool_id)
{
  atomic_store(&(task_pool_lock[pool_id]), false);
}

/*---------------------------------------------------------------------------*/

/* Task_pop() grabs a task from a pool and stores it in the task
   argument. If the pool is empty, then NULL_TASK is assigned to task. */
void Task_pop(__local Task *task, __global Task *task_pool, __global atomic_int *task_pool_lock, __global int *task_pool_head, const int task_pool_size, int local_id, int pool_id)
{
  if (local_id == 0) {
    *task = NULL_TASK;
    /* spinwait on the pool lock */
    while (!(try_lock(task_pool_lock, pool_id)));
    /* If pool is not empty, pick up the latest inserted task. */
    if (task_pool_head[pool_id] > 0) {
      task_pool_head[pool_id]--;
      *task = task_pool[(task_pool_size * pool_id) +  task_pool_head[pool_id]];
    }
    unlock(task_pool_lock, pool_id);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

/*---------------------------------------------------------------------------*/

/* Task_push() adds the task argument to the indicated pool. If the pool
 is full, then the task argument is left untouched, otherwise NULL_TASK
 is assigned to it. */
void Task_push(__local Task *task, __global Task *task_pool, __global atomic_int *task_pool_lock, __global int *task_pool_head, const int task_pool_size, int local_id, int pool_id)
{
  if (local_id == 0) {
    /* spinwait on the pool lock */
    while (!(try_lock(task_pool_lock, pool_id)));
    /* If pool is not full, insert task */
    if (task_pool_head[pool_id] < task_pool_size) {
      task_pool[(task_pool_size * pool_id) +  task_pool_head[pool_id]] = *task;
      task_pool_head[pool_id]++;
      *task = NULL_TASK;
    }
    unlock(task_pool_lock, pool_id);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

/*---------------------------------------------------------------------------*/

__kernel void
connect_four(
             __global uchar *base_board,
             __global Node *nodes,
             __global atomic_int *node_head,
             __global Task *task_pool,
             __global atomic_int *task_pool_lock,
             __global int *task_pool_head,
             const int num_task_pool,
             const int task_pool_size,
             /* for debugging */
             __global int *debug_int,
             __global uchar *debug_board
             )
{
  __local int val[256];
  __local uchar board[NUM_CELL];
  __local Task task;
  __local int child_id[7];
  /* Moves are stored in uchar since possible values are in 0..6 */
  uchar moves[8];

  int local_id = get_local_id(0);
  int local_size = get_local_size(0);
  int group_id = get_group_id(0);
  int global_id = get_global_id(0);

  /* /\* for debug, compuate base board value *\/ */
  /* if (group_id == 0) { */
  /*   if (local_id == 0) { */
  /*     for (int i = 0; i < NUM_CELL; i++) { */
  /*       board[i] = base_board[i]; */
  /*     } */
  /*   } */
  /*   barrier(CLK_LOCAL_MEM_FENCE); */
  /*   int v = board_value(&board, val, local_id, local_size); */
  /*   if (local_id == 0) { */
  /*     *base_value = v; */
  /*   } */
  /*   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); */
  /* } */

  /* init */
  if (global_id == 0) {
    /* Initiate with the 7 nodes of the first level */
    for (int i = 0; i < 7; i++) {
      /* create node */
      nodes[i].parent = -1;
      nodes[i].level = 1;
      nodes[i].moves[0] = i;
      atomic_store(&(nodes[i].value), 0);
      atomic_store(&(nodes[i].num_child_answer), 0);

      /* register task */
      int pool_id = i % num_task_pool;
      task_pool[(pool_id * task_pool_size)] = i;
      task_pool_head[pool_id] = 1;
    }
    atomic_store(node_head, 7);
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  /* poor man's global barrier, to replace with proper global_barrier()
     for kernel_merge usage */
  while (atomic_load(node_head) != 7);

  barrier(CLK_GLOBAL_MEM_FENCE);

  /* main loop */
  while (true) {

    int pool_id = group_id % num_task_pool;

    Task_pop(&task, task_pool, task_pool_lock, task_pool_head, task_pool_size, local_id, group_id);
    if (task == NULL_TASK) {
      break;
    }
    /* compute value of node */
    int value = compute_node_value(base_board, board, val, &(nodes[task]), local_id, local_size);

    if (task == 35 && local_id == 0) {
      *debug_int = value;
      for (int i = 0; i < NUM_CELL; i++) {
        debug_board[i] = board[i];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (nodes[task].level < 2) {
      if (value != PLUS_INF && value != MINUS_INF) {
        create_children(child_id, nodes, node_head, task, local_id, local_size);
        for (int i = 0; i < 7; i++) {
          task = child_id[i];
          /* fixme: task_push may fail, in which case use task_donate once
             it is implemented */
          Task_push(&task, task_pool, task_pool_lock, task_pool_head, task_pool_size, local_id, pool_id);
        }
      } else {
        /* todo: update parent without creating children */
        ;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
