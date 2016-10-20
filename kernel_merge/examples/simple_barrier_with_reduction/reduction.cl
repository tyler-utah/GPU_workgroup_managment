kernel void MY_reduce(
            int length,
            __global int* buffer,
            __global atomic_int* result) {

  __local int scratch[256];
  int gid = get_global_id(0);
  int local_index = get_local_id(0);
  int stride = get_global_size(0);
  
  for (int global_index = gid; global_index < length; global_index += stride) {
    if (global_index < length) {
      scratch[local_index] = buffer[global_index];
    } else {
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
    if (local_index == 0) {
	  atomic_fetch_min((result), scratch[0]);
    }
  }
}
