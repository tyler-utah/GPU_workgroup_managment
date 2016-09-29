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

#include "task.h"
#include "rand.h"

struct DequeHeader
{
	volatile int tail;
	volatile int head;
};

class LBABP;

class DLBABP
{
public:
	friend LBABP;
	Task* deq;
	DequeHeader* dh;
	unsigned int maxlength;
	__device__ int pop(Task& val);
	__device__ int steal(Task& val, unsigned int idx);
	__device__ void push(Task& val);
	__device__ int dequeue2(Task& val);
public:
	__device__ void enqueue(Task& val);
	__device__  int dequeue(Task& val);
};

class LBABP
{
	bool init;
	DLBABP* wq;
	DLBABP* dwq;
public:
	LBABP():init(false){}
	~LBABP();
	int getMaxMem();
	int blocksleft() {return 0;}
	bool setQueueSize(unsigned int dequelength, unsigned int blocks);
	DLBABP* deviceptr() {return dwq;}
};

__device__ int maxl=0;

__device__ void DLBABP::push(Task& val)
{
	deq[blockIdx.x*maxlength+dh[blockIdx.x].tail] = val;
	dh[blockIdx.x].tail++;

	if(maxl<dh[blockIdx.x].tail)
		atomicMax(&maxl,dh[blockIdx.x].tail);
}

__device__ void DLBABP::enqueue(Task& val)
{
	if(threadIdx.x==0)
	{
		push(val);
	}
}

__device__ int getIndex(int head)
{
	return head&0xffff;
}

__device__ int getZeroIndexIncCtr(int head)
{
	return (head+0x10000)&0xffff0000;
}

__device__ int incIndex(int head)
{
	return head+1;
}


__device__ int DLBABP::steal(Task& val, unsigned int idx)
{
	int localTail;
	int oldHead;
	int newHead;

	oldHead = dh[idx].head;
	localTail = dh[idx].tail;
	if(localTail<=getIndex(oldHead))
		return -1;

	val = deq[idx*maxlength+getIndex(oldHead)];
	newHead = incIndex(oldHead);
	if(atomicCAS((int*)&(dh[idx].head),oldHead,newHead)==oldHead)
		return 1;

	return -1;
}

__device__ int DLBABP::pop(Task& val)
{
	int localTail;
	int oldHead;
	int newHead;

	localTail = dh[blockIdx.x].tail;
	if(localTail==0)
		return -1;
	
	localTail--;

	dh[blockIdx.x].tail=localTail;

	val = deq[blockIdx.x*maxlength+localTail];

	oldHead = dh[blockIdx.x].head;

	if (localTail > getIndex(oldHead))
	{
		return 1;
	}

	dh[blockIdx.x].tail = 0;
	newHead = getZeroIndexIncCtr(oldHead);
	if(localTail == getIndex(oldHead))
		if(atomicCAS((int*)&(dh[blockIdx.x].head), oldHead, newHead)==oldHead)
			return 1;
	dh[blockIdx.x].head=newHead;
	return -1;
}

__device__ int DLBABP::dequeue2(Task& val)
{
	if(pop(val)==1)
		return 1;

	if(steal(val,myrand()%gridDim.x)==1)
		return 1;
	else return 0;
}

__device__ int DLBABP::dequeue(Task& val)
{
	__shared__ volatile int rval;
	int dval=0;

	if(threadIdx.x==0)
	{
		rval = dequeue2(val);
	}
	__syncthreads();
	dval = rval;
	__syncthreads();

	return dval;
}
