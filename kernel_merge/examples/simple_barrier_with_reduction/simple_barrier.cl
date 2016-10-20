__kernel void simple_barrier() {
  int i = 0;
  while (true) {
    i++;
    resizing_global_barrier();
    if (i == 100000) {
      break;
    }
  }
}