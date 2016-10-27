
// Simple min reduce from:
// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
__kernel void MY_reduce(
            __global int* buffer,
            int length,
            __global atomic_int* result) {
				
  __local int scratch[256];
  int gid = get_global_id(0);
  int local_index = get_local_id(0);
  int stride = get_global_size(0);
  
  for (int global_index = gid; global_index < length; global_index += stride) {
    // Load data into local memory
    if (global_index < length) {
      scratch[local_index] = buffer[global_index];
    } else {
      // Infinity is the identity element for the min operation
      scratch[local_index] = INT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
	
    for(int offset = 1;
        offset < get_local_size(0);
        offset <<= 1) {
      int mask = (offset << 1) - 1;
      if ((local_index & mask) == 0) {
        int other = scratch[local_index + offset];
        int mine = scratch[local_index];
        scratch[local_index] = (mine < other) ? mine : other;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
	// Putting an atomic here so we can get a global reduction
    if (local_index == 0) {
	  atomic_fetch_min((result), scratch[0]);
    }
  }
}
//
