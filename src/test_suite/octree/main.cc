#include <stdio.h>

#include "opencl/opencl.h"
#include "test_suite/octree/octree_host_dev_shared.h"
#include "test_suite/octree/lbabp.h"
#include "test_suite/octree/octree.h"

/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[]) {

  if (argc!=6) {
    printf("\nUsage:\t./octreepart threads blocks [abp|static] particleCount maxChildren\n\n");
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

  int particleCount = atoi(argv[4]);
  if (particleCount<=0||particleCount>50000000) {
    printf("particleCount must be between 1 and 5000000\n");
    return 1;
  }

  int maxChildren = atoi(argv[5]);
  if (maxChildren<=0||maxChildren>100) {
    printf("maxChildren must be between 1 and 100\n");
    return 1;
  }

  LBMethod method;

  if (!strcmp(argv[3],"abp")) {
    method = Dynamic;
  } else {
    printf ("OpenCL version only support work-stealing method (aka 'abp')\n");
    return 1;
  }
    // if(!strcmp(argv[3],"static"))
    //   method=Static;
    // else {
    //   printf("Load balancing method needs to be either 'abp' or 'static'\n");
    //   return 1;
    // }

  Octree o;
  o.run(threads, blocks, method, maxChildren, particleCount);

  printf("Threads: %d Blocks: %d Method: %s ParticleCount: %d maxChildren: %d "
         "MaxMem: %d Time: %f\n",
         threads, blocks, argv[3], particleCount, maxChildren,
         o.getMaxMem(), o.getTime());
  float err = o.printStats();

  return 0;
}
