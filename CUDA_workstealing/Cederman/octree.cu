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

#include "octree.h"
#include "helper.h"
#include "octree_kernel.h"
#include "stdio.h"

float Octree::printStats()
{
	unsigned int* htree = new unsigned int[MAXTREESIZE];
	unsigned int htreeSize;
	unsigned int hparticlesDone;

	CUDA_SAFE_CALL(cudaMemcpy(&hparticlesDone,particlesDone,sizeof(unsigned int),cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaMemcpy(&htreeSize,treeSize,sizeof(unsigned int),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(htree,tree,sizeof(unsigned int)*MAXTREESIZE,cudaMemcpyDeviceToHost));

	if(htreeSize>=MAXTREESIZE)
	{
		printf("Tree to large!\n");
		return -1;
	}

	unsigned int sum = 0;
	for(unsigned int i=0;i<htreeSize;i++)
	{
	  //printf("%d\n",htree[i]);
		if(htree[i]&0x80000000)
		{
			sum+=htree[i]&0x7fffffff;
		}
	}

	printf("Tree size: %d\n",htreeSize);
	printf("Particles in tree: %d (%d) [%d]\n",sum,numParticles,hparticlesDone);

	float numf = numParticles;
	float sumf = sum;

	delete htree;
	return (numf - sumf)/numf;
}

bool Octree::run(unsigned int threads, unsigned int blocks, LBMethod method, int maxChildren, int numParticles)
{
	this->method = method;
	this->numParticles = numParticles;

	CUDA_SAFE_CALL(cudaMalloc((void**)&tree,sizeof(unsigned int)*MAXTREESIZE));
	CUDA_SAFE_CALL(cudaMemset(tree, 0, sizeof(unsigned int)*MAXTREESIZE));
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles,sizeof(float4)*numParticles));
	CUDA_SAFE_CALL(cudaMalloc((void**)&newParticles,sizeof(float4)*numParticles));
	CUDA_SAFE_CALL(cudaMalloc((void**)&particlesDone,sizeof(unsigned int)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&treeSize,sizeof(unsigned int)));


	generateParticles();

	if(method==Dynamic)
		lbws.setQueueSize(256,blocks);
	else
		if(method==Static)
			lbstat.setQueueSize(900000,blocks);

	if(method == Dynamic)
		initOctree<DLBABP><<<1,1>>>(lbws.deviceptr(),treeSize,particlesDone,numParticles);

	if(method == Static)
		initOctree<DLBStatic><<<1,1>>>(lbstat.deviceptr(),treeSize,particlesDone,numParticles);

	CUT_CHECK_ERROR("initOctree failed!\n");

	Time timer(1);
	timer.start();

	if(method == Dynamic) {          
          makeOctree<DLBABP><<<blocks,threads>>>(lbws.deviceptr(),particles,newParticles,tree,numParticles,treeSize,particlesDone,maxChildren,false);
	}
	else
		if(method == Static)
		{
			while((lbstat.blocksleft())!=0)
			{

				makeOctree<DLBStatic><<<blocks,threads>>>(lbstat.deviceptr(),particles,newParticles,tree,numParticles,treeSize,particlesDone,maxChildren,true);
			}
		}

		CUT_CHECK_ERROR("makeOctree failed!\n");

		float time = timer.stop();

		totalTime = time;

		CUDA_SAFE_CALL(cudaFree(newParticles));
		return true;
}

double genrand_real1(void);
void Octree::generateParticles()
{
	float4* lparticles = new float4[numParticles];

	char fname[256];
	sprintf(fname,"octreecacheddata-%dparticles.dat",numParticles);
	FILE* f = fopen(fname,"rb");
	if(!f)
	{
		printf("Generating and caching data\n");

		int clustersize = 100;
		for(unsigned int i=0;i<numParticles/clustersize;i++)
		{
			float x = ((float)genrand_real1()*800.0f-400.0f);
			float y = ((float)genrand_real1()*800.0f-400.0f);
			float z = ((float)genrand_real1()*800.0f-400.0f);

			for(int x=0;x<clustersize;x++)
			{	
				lparticles[i*clustersize+x].x = x + ((float)genrand_real1()*100.0f-50.0f);
				lparticles[i*clustersize+x].y = y + ((float)genrand_real1()*100.0f-50.0f);
				lparticles[i*clustersize+x].z = z + ((float)genrand_real1()*100.0f-50.0f);

			}

		}

		FILE* f = fopen(fname,"wb");
		fwrite(lparticles,sizeof(float4),numParticles,f);
		fclose(f);
	}
	else
	{
		fread(lparticles,sizeof(float4),numParticles,f);
		fclose(f);
	}

	CUDA_SAFE_CALL(cudaMemcpy(particles,lparticles,sizeof(float4)*numParticles,cudaMemcpyHostToDevice));
	delete lparticles;
}

float Octree::getTime()
{
	return totalTime;
}

int Octree::getMaxMem()
{
	if(method==Dynamic)
		return lbws.getMaxMem();
	else
		if(method == Static)
			return lbstat.getMaxMem();

	return -1;
}
