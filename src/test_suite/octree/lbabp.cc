#include <stdio.h>

#include "opencl/opencl.h"
#include "test_suite/octree/shared_host_side.h"
#include "test_suite/octree/lbabp.h"
#include "test_suite/octree/helper.h"

/*---------------------------------------------------------------------------*/

// // LBABP::~LBABP()
// {
//   //if(init) {
//     // Hugues: do we need to delete buffers in opencl ?
//     // this used to be cudaFree() calls...
//     // delete(&dwq);
//     // delete(wq->deq);
//     // delete(wq->dh);
//   //}
// }

/*---------------------------------------------------------------------------*/

bool LBABP::setQueueSize(cl::Context context, cl::CommandQueue queue, cl::Program program, unsigned int dequelength, unsigned int blocks)
{
  init = true;

  // Hugues: Cuda version builds and fill a host-side 'wq' and then
  // copies it to device, but 'wq' contains pointers... In OpenCL, I
  // guess we can edit these pointers only through a kernel
  // call. Therefore I create all buffers here, and call the
  // initDLBABP() kernel (which does not exists in Cuda) to set the
  // pointers on the device side.

  maxl = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int));
  dwq = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(DLBABP));
  deq = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(Task) * dequelength * blocks);
  queue.enqueueFillBuffer(deq, 0, 0, sizeof(Task) * dequelength * blocks, NULL, NULL);
  dh = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(DequeHeader) * blocks);
  queue.enqueueFillBuffer(dh, 0, 0, sizeof(DequeHeader) * blocks, NULL, NULL);
  maxlength = dequelength;

  cl_int err;
  cl::Kernel kernel(program, "initDLBABP", &err);
  checkErr(err, "kernel constructor for initDLBABP");
  kernel.setArg(0, dwq);
  kernel.setArg(1, deq);
  kernel.setArg(2, dh);
  kernel.setArg(3, maxlength);
  cl::Event event;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1), NULL, &event);
  event.wait();

  return true;
}

/*---------------------------------------------------------------------------*/

// Hugues: the retrievial of maxl is now done at the end of Octree::run()

// int LBABP::getMaxMem(cl::CommandQueue queue)
// {
//   int maxle = 0;

//   queue.enqueueReadBuffer(maxl, CL_TRUE, 0, sizeof(int), &maxle, NULL, NULL);

//   return maxle;
// }

/*---------------------------------------------------------------------------*/
