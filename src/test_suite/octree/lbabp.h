// requires:
// #include "octree_host_dev_shared.h"

/*---------------------------------------------------------------------------*/

class LBABP
{
  bool init;
  /* fields of DLBABP */
  /* The device buffer for DLBABP */
  cl::Buffer dwq;
  /* This is a kernel global in Cuda */
  cl::Buffer maxl;

public:
  cl::Buffer deq;
  cl::Buffer dh;
  unsigned int maxlength;
  cl::Buffer getMaxl() {return maxl;}
  LBABP():init(false){}
  //~LBABP();
  //int getMaxMem(cl::CommandQueue queue);
  int blocksleft() {return 0;}
  bool setQueueSize(cl::Context context, cl::CommandQueue queue, cl::Program program, unsigned int dequelength, unsigned int blocks);
  cl::Buffer deviceptr() {return dwq;}
};

/*---------------------------------------------------------------------------*/
