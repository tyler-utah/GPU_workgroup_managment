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

#include "task.h"
#include "lbstatic.h"
#include "helper.h"

LBStatic::~LBStatic()
{
	if(init)
	{
		cudaFree(dwq);
		cudaFree(wq->indeq);
		cudaFree(wq->outdeq);
		free(wq);
	}
}

bool LBStatic::setQueueSize(unsigned int dequelength, unsigned int blocks)
{
	init = true;
	this->blocks = blocks;
	wq = (DLBStatic*)malloc(sizeof(DLBStatic));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dwq,sizeof(DLBStatic)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&(wq->indeq),sizeof(Task)*dequelength));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(wq->outdeq),sizeof(Task)*dequelength));

	CUDA_SAFE_CALL(cudaMalloc((void**)&(wq->ctrs),sizeof(unsigned int)*blocks));

	CUDA_SAFE_CALL(cudaMemset(wq->ctrs,0,sizeof(unsigned int)*blocks));

	wq->ctr=0;
	wq->ctr2=0;

	CUDA_SAFE_CALL(cudaMemcpy(dwq,wq,sizeof(DLBStatic),cudaMemcpyHostToDevice));

	return true;
}

int LBStatic::getMaxMem()
{
	return smaxl;
}

unsigned int LBStatic::blocksleft()
{
	CUDA_SAFE_CALL(cudaMemcpy(wq,dwq,sizeof(DLBStatic),cudaMemcpyDeviceToHost));
	if(wq->ctr==0)
		return 0;

	Task* t = wq->indeq;
	wq->indeq = wq->outdeq;
	wq->outdeq = t;

	int rval = wq->ctr;
	wq->ctr = 0;
	wq->ctr2 = rval;

	CUDA_SAFE_CALL(cudaMemcpy(dwq,wq,sizeof(DLBStatic),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemset(wq->ctrs,0,sizeof(unsigned int)*blocks));

	if(smaxl<(int)wq->ctr2)
		smaxl=(int)wq->ctr2;

	return rval;
}
