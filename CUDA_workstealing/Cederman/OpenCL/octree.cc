#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <streambuf>

#include <CL/cl.hpp>
#include "octree_host_dev_shared.h"
#include "lbabp.h"
#include "octree.h"
#include "helper.h"

/*---------------------------------------------------------------------------*/

float Octree::printStats()
{
  // Hugues: all stats are retrieved from device memory at the end of
  // Octree::run().

  if(htreeSize >= MAXTREESIZE) {
    printf("Tree too large!\n");
    return -1;
  }

  unsigned int sum = 0;
  for(unsigned int i = 0; i < htreeSize; i++) {
    //printf("%d\n",htree[i]);
    if (htree[i] & 0x80000000) {
      sum += htree[i] & 0x7fffffff;
    }
  }

  printf("Tree size: %d\n", htreeSize);
  printf("Particles in tree: %d (%d) [%d]\n", sum, numParticles, hparticlesDone);

  float numf = numParticles;
  float sumf = sum;

  return (numf - sumf) / numf;
}

/*---------------------------------------------------------------------------*/

bool Octree::run(unsigned int threads, unsigned int blocks, LBMethod method, int maxChildren, int numParticles)
{
  this->method = method;
  this->numParticles = numParticles;

  // Initiate OpenCL
  cl_int err;
  cl::Event event;

  // platform
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    printf("Platform size is 0\n");
    exit (EXIT_FAILURE);
  }

  // context
  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
  cl::Context context(CL_DEVICE_TYPE_GPU, properties);

  // device
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

  // command queue
  cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
  checkErr(err, "cl::CommandQueue::CommandQueue()");

  // Now we can create device memory buffers

  // Hugues TODO: check if something more restrictive than CL_MEM_READ_WRITE
  // would be enough

  tree = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int)*MAXTREESIZE);
  queue.enqueueFillBuffer(tree, 0, 0, sizeof(unsigned int)*MAXTREESIZE, NULL, &event);
  event.wait();
  // event.wait() is not always needed since enqueued workloads are
  // performed in order (this is a queue after all). we need to wait()
  // if we gonna read results in host afterward
  particles = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4)*numParticles);
  newParticles = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4)*numParticles);
  particlesDone = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int));
  treeSize = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int));

  generateParticles(queue);

  //--------------------------------------------------
  // this random data array is in rand.h in the CUDA version
  cl::Buffer randdata(context, CL_MEM_READ_WRITE, sizeof(int)*128);

  //--------------------------------------------------
  // create kernel from file

  // Hugues: something goes wrong with creating kernel from file ?? YES,
  // there are 2 OpenCL on hemlock machine, and by default the program
  // loaded /usr/lib/x86_64-linux-gnu/libOpenCL.so, which is buggy to
  // the point that it segfaults at load time if the programs refers to
  // std::ostringstream (just decalring a variable of this type leads to
  // a segfault ! ).

  // We use the LD_PRELOAD trick to load
  // /vol/cuda/7.5.18/lib64/libOpenCL.so beforehand

  std::ifstream kernelFile("octree_kernel.cl");
  std::ostringstream kernelOss;
  kernelOss << kernelFile.rdbuf();

  cl::Program program(context, kernelOss.str());
  err = program.build(devices);
  if (err != CL_SUCCESS) {
    std::string buildlog;
    buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0], &err);
    std::cout << "Build failed with log:\n" << buildlog << "\n" ;
    exit(EXIT_FAILURE);
  }

  //--------------------------------------------------
  // In OpenCL, only dynamic method is implemented

  lbws.setQueueSize(context, queue, program, 256, blocks);

  cl::Kernel kernel(program, "initOctree", &err);
  checkErr(err, "kernel constructor for initOctree");
  kernel.setArg(0, lbws.deviceptr());
  kernel.setArg(1, lbws.getMaxl());
  kernel.setArg(2, treeSize);
  kernel.setArg(3, particlesDone);
  kernel.setArg(4, numParticles);

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1), NULL, &event);
  event.wait();

  kernel = cl::Kernel(program, "makeOctree", &err);
  checkErr(err, "kernel constructor for makeOctree");
  kernel.setArg(0, lbws.deviceptr());
  kernel.setArg(1, randdata);
  kernel.setArg(2, lbws.getMaxl());
  kernel.setArg(3, particles);
  kernel.setArg(4, newParticles);
  kernel.setArg(5, tree);
  kernel.setArg(6, numParticles);
  kernel.setArg(7, treeSize);
  kernel.setArg(8, particlesDone);
  kernel.setArg(9, maxChildren);
  kernel.setArg(10, false);

  cl::NDRange local_size(threads);
  cl::NDRange global_size(blocks);

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, NULL, &event);
  event.wait();

  // ----- gather stats -----

  // execution time
  cl_ulong start_time, end_time;
  err = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
  checkErr(err, "getProfilingInfo() start");
  event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
  checkErr(err, "getProfilingInfo() end");

  // Hugues: OpenCL gives time in nanosec, convert it to millisec to
  // match CUDA timer
  totalTime = ((float)(end_time - start_time) / (float)1000000);

  // Hugues: in Cuda version, maxl is set within LBABP::getMaxMem(), but
  // that would need to pass the queue around. I thinks this is
  // overkill, we just gather all stats here.
  queue.enqueueReadBuffer(lbws.getMaxl(), CL_TRUE, 0, sizeof(int), &maxMem, NULL, NULL);

  // Hugues: the Cuda code corresponding to the following is in
  // Octree::printStats()
  queue.enqueueReadBuffer(particlesDone, CL_TRUE, 0, sizeof(unsigned int), &hparticlesDone, NULL, NULL);
  queue.enqueueReadBuffer(treeSize, CL_TRUE, 0, sizeof(unsigned int), &htreeSize, NULL, NULL);
  htree = new unsigned int[MAXTREESIZE];
  queue.enqueueReadBuffer(tree, CL_TRUE, 0, sizeof(unsigned int) * MAXTREESIZE, htree, NULL, NULL);

  // Hugues: do we need to free things in OpenCL ? cl::Buffers are
  // collected by the garbage collector of c++ runtime ?

  // 	CUDA_SAFE_CALL(cudaFree(newParticles));

  return true;
}

/*---------------------------------------------------------------------------*/

double genrand_real1(void);
void Octree::generateParticles(cl::CommandQueue queue)
{
  cl_float4* lparticles = new cl_float4[numParticles];

  char fname[256];
  sprintf(fname,"octreecacheddata-%dparticles.dat",numParticles);
  FILE* f = fopen(fname,"rb");
  if (!f) {
    printf("Generating and caching data\n");

    int clustersize = 100;
    for (unsigned int i=0; i<numParticles/clustersize; i++) {
      float x = ((float)genrand_real1()*800.0f-400.0f);
      float y = ((float)genrand_real1()*800.0f-400.0f);
      float z = ((float)genrand_real1()*800.0f-400.0f);

      for (int x=0;x<clustersize;x++) {
        lparticles[i*clustersize+x].x = x + ((float)genrand_real1()*100.0f-50.0f);
        lparticles[i*clustersize+x].y = y + ((float)genrand_real1()*100.0f-50.0f);
        lparticles[i*clustersize+x].z = z + ((float)genrand_real1()*100.0f-50.0f);
      }

    }

    FILE* f = fopen(fname,"wb");
    fwrite(lparticles,sizeof(cl_float4),numParticles,f);
    fclose(f);
  } else {
    fread(lparticles,sizeof(cl_float4),numParticles,f);
    fclose(f);
  }

  cl::Event event;
  queue.enqueueWriteBuffer(particles,
                           CL_TRUE, // blocking write since we
                           // delete lparticles afterward
                           0,
                           sizeof(cl_float4)*numParticles,
                           lparticles,
                           NULL,
                           &event);
  event.wait();

  delete lparticles;
}

/*---------------------------------------------------------------------------*/

float Octree::getTime()
{
  return totalTime;
}

/*---------------------------------------------------------------------------*/

int Octree::getMaxMem()
{
  return maxMem;
}

/*---------------------------------------------------------------------------*/
