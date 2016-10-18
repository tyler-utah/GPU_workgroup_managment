#include <stdio.h>

#include "opencl/opencl.h"
#include "test_suite/octree/shared_host_side.h"
#include "test_suite/octree/lbabp.h"
#include "test_suite/octree/octree.h"

/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[]) {

  if (argc!=5) {
    printf("\nUsage:\t./octreepart threads blocks particleCount maxChildren\n\n");
    return 1;
  }

  int threads = atoi(argv[1]);
  if (threads<=0||threads>512) {
    printf("Threads must be between 1 and 128\n");
    return 1;
  }

  int blocks = atoi(argv[2]);
  if (blocks<=0||blocks>512) {
    printf("Blocks must be between 1 and 512\n");
    return 1;
  }

  int particleCount = atoi(argv[3]);
  if (particleCount<=0||particleCount>50000000) {
    printf("particleCount must be between 1 and 5000000\n");
    return 1;
  }

  int maxChildren = atoi(argv[4]);
  if (maxChildren<=0||maxChildren>100) {
    printf("maxChildren must be between 1 and 100\n");
    return 1;
  }

  LBMethod method = Dynamic;

  Octree o;
  o.run(threads, blocks, method, maxChildren, particleCount);

  printf("Threads: %d Blocks: %d Method: dynamic ParticleCount: %d maxChildren: %d "
         "MaxMem: %d Time: %f\n",
         threads, blocks, particleCount, maxChildren,
         o.getMaxMem(), o.getTime());
  float err = o.printStats();

  return 0;
}
