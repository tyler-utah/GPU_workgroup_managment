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

#pragma once
class LBStatic;

class DLBStatic
{
public:
	friend LBStatic;
	Task* indeq;
	Task* outdeq;
	unsigned int ctr;
	unsigned int ctr2;
	unsigned int* ctrs;
public:
	__device__ void enqueue(Task& val);
	__device__ int dequeue(Task& val);
};


class LBStatic
{
	bool init;
	DLBStatic* wq;
	DLBStatic* dwq;
	int blocks;
	int smaxl;
public:
	LBStatic():init(false),smaxl(0){};
	~LBStatic();
	int getMaxMem();
	bool setQueueSize(unsigned int dequelength, unsigned int blocks);
	DLBStatic* deviceptr() {return dwq;}
	unsigned int blocksleft();
};


__device__ void DLBStatic::enqueue(Task& val)
{
	if(threadIdx.x==0)
	{
		int pos = atomicAdd(&ctr,1);
		indeq[pos]=val;
	}
}

__device__ int DLBStatic::dequeue(Task& t)
{
	__shared__ volatile int rval;
	int dval;

	if(threadIdx.x==0)
	{
		if(blockIdx.x+ctrs[blockIdx.x]*gridDim.x<ctr2)
		{
			t = outdeq[blockIdx.x+ctrs[blockIdx.x]*gridDim.x];
			ctrs[blockIdx.x]++;
			rval = 1;
		}
		else
			rval = 0;
	}
	__syncthreads();
	dval = rval;
	__syncthreads();
	return dval;
}
