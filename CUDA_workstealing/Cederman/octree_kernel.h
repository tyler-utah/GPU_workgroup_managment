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
#include "lbstatic.h"
#include "lbabp.h"

__constant__ __device__  int mc[8][3] = 
{{-1,-1,-1},{+1,-1,-1},{-1,+1,-1},{+1,+1,-1},
 {-1,-1,+1},{+1,-1,+1},{-1,+1,+1},{+1,+1,+1}};

__device__ int whichbox(volatile float4& pos, float4& middle)
{
	int box = pos.x<middle.x?0:1;
	box    += pos.y<middle.y?0:2;
	box    += pos.z<middle.z?0:4;
	return box;
}
/*
0   - - -
1   + - -
2   - + -
3   + + -
4   - - +
5   + - +
6   - + +
7   + + +
*/


template <typename Work>
__global__ void makeOctree(Work* workpool, float4* particles, float4* newparticles, unsigned int* tree,
						  unsigned int particleCount,unsigned int* treeSize,unsigned int* particlesDone,
						 unsigned int maxchilds, bool isStatic)
{

	__shared__ float4* frompart;
	__shared__ float4* topart;

	__shared__ unsigned int count[8];
	__shared__ int sum[8];
	

	__shared__ Task t;
	__shared__ unsigned int check;

	while(true)
	{
		__syncthreads();


		// Try to acquire new task
		if(workpool->dequeue(t)==0)
		{
			if(isStatic)
				return;
			check = *particlesDone;
			__syncthreads();
			if(check==particleCount) 
				return;
			continue;
		}

			if(t.flip)
			{
				frompart = newparticles;
				topart = particles;
			}
			else
			{
				frompart = particles;
				topart = newparticles;
			}

		__syncthreads();
		
		for(int i=threadIdx.x;i<8;i+=blockDim.x)
		{
			count[i]=0;
		}



		__syncthreads();

		for(int i=t.beg+threadIdx.x;i<t.end;i+=blockDim.x)
		{
			atomicAdd(&count[whichbox(frompart[i],t.middle)],1);
		}
		__syncthreads();

		if(threadIdx.x==0)
		{
			sum[0]=count[0];
			for(int x=1;x<8;x++)
				sum[x]=sum[x-1]+count[x];
		}

		__syncthreads();

		for(unsigned int i=t.beg+threadIdx.x;i<t.end;i+=blockDim.x)
		{
			int toidx = t.beg+atomicAdd(&sum[whichbox(frompart[i],t.middle)],-1)-1;
			topart[toidx]=frompart[i];
		}
		__syncthreads();

		for(int i=0;i<8;i+=1)
		{
			Task newTask;

			// Create new work or move to correct side
			if(count[i]>maxchilds)
			{
				if(threadIdx.x==0)
				{
					newTask.middle.x = t.middle.x+t.middle.w*mc[i][0];
					newTask.middle.y = t.middle.y+t.middle.w*mc[i][1];
					newTask.middle.z = t.middle.z+t.middle.w*mc[i][2];
					newTask.middle.w = t.middle.w/2.0;

					newTask.flip = !t.flip;
					newTask.beg = t.beg+sum[i];
					newTask.end = newTask.beg+count[i];

					tree[t.treepos+i]=atomicAdd(treeSize,(unsigned int)8);
					newTask.treepos=tree[t.treepos+i];
					workpool->enqueue(newTask);
				}
			}
			else
			{
				if(!t.flip)
				{
					for(int j=t.beg+sum[i]+threadIdx.x;j<t.beg+sum[i]+count[i];j+=blockDim.x)
						particles[j]=topart[j];
				}
				__syncthreads();
				if(threadIdx.x==0)
				{
					atomicAdd(particlesDone,count[i]);
					unsigned int val = count[i];
					tree[t.treepos+i]=0x80000000|val;
				}
			}
		}
	}
}

template <typename Work>
__global__ void initOctree(Work* workpool, unsigned int* treeSize, unsigned int* particlesDone, int numParticles)
{
	*treeSize = 100;
	*particlesDone = 0;

	Task t;
	t.treepos=0;
	t.middle.x=0;
	t.middle.y=0;
	t.middle.z=0;
	t.middle.w=256;

	t.beg = 0;
	t.end = numParticles;
	t.flip = false;

	workpool->enqueue(t);
}
