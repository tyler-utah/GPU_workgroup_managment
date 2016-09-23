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

#include "lbabp.h"
#include "helper.h"


LBABP::~LBABP()
{
	if(init)
	{
		cudaFree(dwq);
		cudaFree(wq->deq);
		cudaFree(wq->dh);
	}

}


bool LBABP::setQueueSize(unsigned int dequelength, unsigned int blocks)
{
	init = true;
	wq = (DLBABP*)malloc(sizeof(DLBABP));

	CUDA_SAFE_CALL(cudaMalloc((void**)&dwq,sizeof(DLBABP)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&(wq->deq),sizeof(Task)*dequelength*blocks));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(wq->dh),sizeof(DequeHeader)*blocks));

	CUDA_SAFE_CALL(cudaMemset(wq->deq,0,sizeof(Task)*dequelength*blocks));
	CUDA_SAFE_CALL(cudaMemset(wq->dh,0,sizeof(DequeHeader)*blocks));

	wq->maxlength = dequelength;
	CUDA_SAFE_CALL(cudaMemcpy(dwq,wq,sizeof(DLBABP),cudaMemcpyHostToDevice));
	return true;
}

int LBABP::getMaxMem()
{
	int maxle = 0;
	//CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&maxle,"maxl",sizeof(int),0,cudaMemcpyDeviceToHost));
	return maxle;
}
