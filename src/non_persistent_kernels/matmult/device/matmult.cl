/*
  Matrix multiplication: C = A * B

  We assume C is big enough to store the result.
*/

/*---------------------------------------------------------------------------*/

__kernel void matmult(
                      __global int *A,
                      const int A_row,
                      const int A_col,
                      __global int *B,
                      const int B_row,
                      const int B_col,
                      __global int *C)
{
  /* safety */
  if (A_col != B_row) {
    return;
  }

  int gid = get_global_id(0);
  int num_threads = get_global_size(0);
  int c_size = A_row * B_col;

  /* Multiply matrices */
  for (int i = gid; i < c_size; i += num_threads) {
    int c_row = i / B_col;
    int c_col = i % B_col;
    int a_offset = c_row * A_col;
    C[i] = 0;
    for (int j = 0; j < B_row; j++) {
      C[i] += A[a_offset + j] * B[(j * B_col) + c_col];
    }
  }
}

/*---------------------------------------------------------------------------*/
