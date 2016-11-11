/*---------------------------------------------------------------------------*/

#define EMPTY ((uchar)0)
#define HUMAN ((uchar)1)
#define COMPUTER ((uchar)2)

/*---------------------------------------------------------------------------*/

/* Warning: there is still some parts of code where these values are
 * hardcoded (in particular, diagonals scanning) */
const int NUM_ROW = 6;
const int NUM_COL = 7;
const int NUM_CELL = 6 * 7;

const int PLUS_INF = 666;
const int MINUS_INF = -666;

/* The actual level of lookahead is controlled by the "maxlevel" flag,
 * MAX_LOOKAHEAD is a hard-coded upper-bound. */
const int MAX_LOOKAHEAD = 7;

/*---------------------------------------------------------------------------*/

typedef struct {
  int parent;
  int level;
  uchar moves[MAX_LOOKAHEAD];
  atomic_int value;
  atomic_int num_child_answer;
} Node;

/*---------------------------------------------------------------------------*/

/* A task is just a int indicating node index */
const int NULL_TASK = -1;

/*---------------------------------------------------------------------------*/

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
int board_value(__local uchar *board, int local_id, int local_size)
{
  int value = 0;

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

    /* update value */
    if (tmpval == PLUS_INF || tmpval == MINUS_INF) {
      /* found a winner */
      value = tmpval;
      break;
    } else {
      value += tmpval;
    }
  }

  return value;
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

/* wgm_create_children() must be called by the workgroup master
   thread. This function returns the node id of the created child */
int wgm_create_children(int move, int parent_id, __global Node *nodes, __global atomic_int *node_head, int num_nodes)
{
  int child_id = NULL_TASK;
  __global Node *parent = &(nodes[parent_id]);
  child_id = atomic_fetch_add(node_head, 1);
  /* safety */
  if (child_id <= num_nodes) {
    Node *child = &(nodes[child_id]);
    child->parent = parent_id;
    child->level = parent->level + 1;
    for (int j = 0; j < parent->level; j++) {
      child->moves[j] = parent->moves[j];
    }
    child->moves[parent->level] = move;
    atomic_store(&(child->value), 0);
    atomic_store(&(child->num_child_answer), 0);
  }
  return child_id;
}

/*---------------------------------------------------------------------------*/

bool pool_try_lock(__global atomic_int *task_pool_lock, int pool_id)
{
  int expected = false;
  return atomic_compare_exchange_strong(&(task_pool_lock[pool_id]), &expected, true);
}

/*---------------------------------------------------------------------------*/

void pool_unlock(__global atomic_int *task_pool_lock, int pool_id)
{
  atomic_store(&(task_pool_lock[pool_id]), false);
}

/*---------------------------------------------------------------------------*/

/* wgm_task_pop() MUST be called by local_id 0. This function grabs a
   task from a pool and stores it in the task argument. If the pool is
   empty, then NULL_TASK is assigned to task. */
int wgm_task_pop(__global int *task_pool, __global atomic_int *task_pool_lock, __global int *task_pool_head, const int task_pool_size, int pool_id)
{
  int task = NULL_TASK;
  /* spinwait on the pool lock */
  while (!(pool_try_lock(task_pool_lock, pool_id)));
  /* If pool is not empty, pick up the latest inserted task. */
  if (task_pool_head[pool_id] > 0) {
    atomic_dec(&(task_pool_head[pool_id]));
    task = task_pool[(task_pool_size * pool_id) +  task_pool_head[pool_id]];
  }
  pool_unlock(task_pool_lock, pool_id);
  return task;
}

/*---------------------------------------------------------------------------*/

/* Task_push() adds the task argument to the indicated pool. If the pool
   is full, then the task argument is left untouched, otherwise
   NULL_TASK is assigned to it. */
int wgm_task_push(int task, __global int *task_pool, __global atomic_int *task_pool_lock, __global int *task_pool_head, const int task_pool_size, int pool_id)
{
  /* spinwait on the pool lock */
  while (!(pool_try_lock(task_pool_lock, pool_id)));
  /* If pool is not full, insert task */
  if (task_pool_head[pool_id] < task_pool_size) {
    task_pool[(task_pool_size * pool_id) +  task_pool_head[pool_id]] = task;
    atomic_inc(&(task_pool_head[pool_id]));
    task = NULL_TASK;
  }
  pool_unlock(task_pool_lock, pool_id);
  return task;
}

/*---------------------------------------------------------------------------*/

/* wgm_update_parent() MUST be called by the workgroup master
   thread. This function propagates the child value to its parent,
   recursively to the top if needed. */
void wgm_update_parent(__global Node *nodes, int node_id, __global int *next_move_value, __global atomic_int *root_done)
{
  while (true) {
    __global Node *node = &(nodes[node_id]);

    if (node->level == 1) {
      /* we've reached the top, a root has terminated */
      next_move_value[node_id] = atomic_load(&(node->value));
      atomic_fetch_add(root_done, 1);
      return;
    }

    __global Node *parent = &(nodes[node->parent]);
    int value = atomic_load(&(node->value));

    if ((parent->level % 2) == 0) {
      /* parent level is even: human takes lowest value */
      atomic_fetch_min(&(parent->value), value);
    } else {
      /* parent level is odd: computer takes highest value */
      atomic_fetch_max(&(parent->value), value);
    }
    /* in all case, increase counter of answer */
    int prev_num_answer = atomic_fetch_add(&(parent->num_child_answer), 1);

    /* if we were the last child to update, move upward */
    if (prev_num_answer == 6) {
      node_id = node->parent;
    } else {
      return;
    }
  }
}

/*---------------------------------------------------------------------------*/

__kernel void
connect_four(
             __global uchar *base_board,
             const int maxlevel,
             __global Node *nodes,
             __global atomic_int *node_head,
             const int num_nodes,
             __global int *task_pool,
             __global atomic_int *task_pool_lock,
             __global int *task_pool_head,
             const int num_task_pool,
             const int task_pool_size,
             __global int *next_move_value,
             __global atomic_int *root_done,
             /* for debugging */
             __global atomic_int *debug_int,
             __global uchar *debug_board
             )
{
  __local int val[256];
  __local uchar board[NUM_CELL];
  /* Hugues: when using kernel_merge we got a strange clang warning if
     we name the __local task variable "task" ... maybe some name
     clashing ? That's why it's called "local_task" */
  __local int local_task[1];
  __local bool game_over[1];

  /* INIT */
  if (get_global_id(0) == 0) {
    /* Initiate with the 7 nodes of the first level */
    for (int i = 0; i < 7; i++) {
      next_move_value[i] = MINUS_INF;
      /* create node */
      nodes[i].parent = -1;
      nodes[i].level = 1;
      nodes[i].moves[0] = i;
      atomic_store(&(nodes[i].value), 0);
      atomic_store(&(nodes[i].num_child_answer), 0);

      /* register task */
      int pool_id = i % num_task_pool;
      task_pool[(pool_id * task_pool_size) + task_pool_head[pool_id]] = i;
      task_pool_head[pool_id] += 1;
    }
    atomic_store(node_head, 7);
    atomic_store(root_done, 0);
  }

  /* BARRIER */
  global_barrier();

  /* MAIN LOOP */
  while (true) {

    if (get_group_id(0) > 0) {
      offer_kill();
    }

    if (get_group_id(0) == 0) {
      offer_fork();
    }

    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int pool_id = get_group_id(0) % num_task_pool;

    /* get task */
    if (local_id == 0) {
      local_task[0] = wgm_task_pop(task_pool, task_pool_lock, task_pool_head, task_pool_size, pool_id);
      /* if no more task in own pool, try to steal some from other pools */
      if (local_task[0] == NULL_TASK) {
        for (int i = 1; i < num_task_pool; i++) {
          int steal_pool_id = (pool_id + i) % num_task_pool;
          local_task[0] = wgm_task_pop(task_pool, task_pool_lock, task_pool_head, task_pool_size, steal_pool_id);
          if (local_task[0] != NULL_TASK) {
            break;
          }
        }
      }
      /* if task is still null, there may be no work left anymore */
      game_over[0] = false;
      if (local_task[0] == NULL_TASK) {
        game_over[0] = (atomic_load(root_done) == NUM_COL);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (local_task[0] == NULL_TASK) {
      if (game_over[0]) {
        break;
      } else {
        continue;
      }
    }

    /* treat task */
    __global Node *node = &(nodes[local_task[0]]);
    update_board(board, base_board, node, local_id, local_size);
    val[local_id] = board_value(board, local_id, local_size);

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    /* Update node value */
    if (local_id == 0) {

      /* Reduce value */
      int value = 0;
      for (int i = 0; i < local_size; i++) {
        if (val[i] == PLUS_INF || val[i] == MINUS_INF) {
          value = val[i];
          break;
        }
        value = value + val[i];
      }
      atomic_store(&(node->value), value);

      if (node->level >= maxlevel ||
        value == PLUS_INF ||
        value == MINUS_INF) {

        /* we reached a leaf */
        atomic_store(&(node->num_child_answer), 7);
        wgm_update_parent(nodes, local_task[0], next_move_value, root_done);

      } else {

        /* create children */
        for (int i = 0; i < NUM_COL; i++) {
          int child_id = wgm_create_children(i, local_task[0], nodes, node_head, num_nodes);
          int push_pool_id = pool_id;
          /* loop on trying to push the task in a pool */
          while (wgm_task_push(child_id, task_pool, task_pool_lock, task_pool_head, task_pool_size, push_pool_id) != NULL_TASK) {
            push_pool_id = (push_pool_id + 1) % num_task_pool;
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}

/*---------------------------------------------------------------------------*/
