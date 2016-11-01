/*
  Matrix multiplication: C = A * B

  We assume C is big enough to store the result.
*/

/*---------------------------------------------------------------------------*/

__kernel void matmult(__global int *A,
                      const int a_num_line,
                      const int a_num_col,
                      __global int *B,
                      const int b_num_line,
                      const int b_num_col,
                      __global int *C,
                      __global ulong *c_hash)
{
  /* safety */
  if (a_num_col != b_num_line) {
    return;
  }

  int gid = get_global_id(0);
  int num_threads = get_global_size(0);
  int c_size = a_num_line * b_num_col;

  for (int i = gid; i < c_size; i += num_threads) {
    int c_line = i / b_num_col;
    int c_col = i % b_num_col;
    int a_offset = c_line * a_num_col;
    C[i] = 0;
    for (int j = 0; j < b_num_line; j++) {
      C[i] += A[a_offset + j] * B[(j * b_num_col) + c_col];
    }
  }

  /* Hash using djb2, see http://www.cse.yorku.ca/~oz/hash.html */
  if (gid == 0) {
    *c_hash = 5381;
    for (int i = 0; i < c_size; i++) {
      *c_hash = (*c_hash) * 33 + C[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
