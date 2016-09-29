/**
 * Octree Partitioning
 * Benchmark for dynamic load balancing using
 * work-stealing on graphics processors.
 * --------------------------------------------------------
 * Copyright (c) 2011, Daniel Cederman and Philippas Tsigas
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following 
 * conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
 * in the documentation and/or other materials provided with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
**/

#include <string.h>
#include <stdio.h>
#include "octree.cu"
#include "helper.h"

int main(int argc, char* argv[])
{

  if(argc!=6)
    {
      printf("\nUsage:\t./octreepart threads blocks [abp|static] particleCount maxChildren\n\n");
      return 1;
    }
  
  int threads = atoi(argv[1]);
  if(threads<=0||threads>512)
    {
      printf("Threads must be between 1 and 128\n");
      return 1;
    }
  
  int blocks = atoi(argv[2]);
  if(blocks<=0||blocks>512)
    {
      printf("Blocks must be between 1 and 512\n");
      return 1;
    }
  
  int particleCount = atoi(argv[4]);
  if(particleCount<=0||particleCount>50000000)
    {
		printf("particleCount must be between 1 and 5000000\n");
		return 1;
    }
  
  int maxChildren = atoi(argv[5]);
  if(maxChildren<=0||maxChildren>100)
    {
      printf("maxChildren must be between 1 and 100\n");
      return 1;
    }
  
  
  LBMethod method;
  
  if(!strcmp(argv[3],"abp"))
    method=Dynamic;
  else
    if(!strcmp(argv[3],"static"))
      method=Static;
    else
      {
	printf("Load balancing method needs to be either 'abp' or 'static'\n");
	return 1;
      }

  
  Octree o;
  o.run(threads,blocks,method,maxChildren,particleCount);
  
  printf("Threads: %d Blocks: %d Method: %s ParticleCount: %d maxChildren: %d MaxMem: %d Time: %f\n",threads,blocks,argv[3],particleCount,maxChildren,o.getMaxMem(),o.getTime());
  float err = o.printStats();


  return 0;
}
