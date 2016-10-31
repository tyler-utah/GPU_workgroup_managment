//This kernel is the naive version of the mega-kernel using
//the global barrier

// Tyler: Notes: Simply adding the scheduler args in the 
// standalone application causes a massive slowdown
// (~13 seconds to ~24 seconds).
// But the merged kernel *necessarily* must have
// the scheduler context, so it isn't really fair
// to compare without it. 
__kernel void sssp_combined( const int num_rows,
                           __global int * row,
                           __global int * col,
                           __global int * data,
                           __global int * x,
                           __global int * y,
                           __global int *stop
                           ) {


  int local_stop = 0;

  while(1) {
	  
	// Get ids
    int tid = get_global_id(0);
    int stride = get_global_size(0);

    // Original application --- vector_assign --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_rows; i+=stride) {
      x[i] = y[i];
    }

    // Original application --- vector_assign --- end

    // Inter-workgroup barrier
    resizing_global_barrier();
	
	// Get ids
    tid = get_global_id(0);
    stride = get_global_size(0);

    // Original application --- spmv_min_dot_plus_kernel --- start

    // This is the right place to initialize stop variable as it is
    // the only barrier interval which doesn't use stop
    *stop = 0;

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int it = tid; it < num_rows; it+=stride) {

      // Get the start and end pointers
      int row_start = row[it];
      int row_end   = row[it+1];

      // Perform + for each pair of elements and a reduction with min
      int min = x[it];
      for (int j = row_start; j < row_end; j++) {
        if (data[j] + x[col[j]] < min)
          min = data[j] + x[col[j]];
      }
      y[it] = min;
    }

    // Original application --- spmv_min_dot_plus_kernel --- end

    // Inter-workgroup barrier
    resizing_global_barrier();
	
	tid = get_global_id(0);
    stride = get_global_size(0);

    // Original application --- vector_diff --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_rows; i+=stride) {
      if (y[i] != x[i])
        *stop = 1;
    }

    // Original application --- vector_diff --- end

    // Inter-workgroup barrier
    resizing_global_barrier();

    // Check terminating condition before continuing
    if (*stop == 0) {
      break;
    }
  }
}
//