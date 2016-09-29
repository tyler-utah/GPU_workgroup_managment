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

#include "lbabp.cu"
#include "lbstatic.cu"


enum LBMethod
{
	Dynamic,
	Static
};

class Octree
{
	static const unsigned int MAXTREESIZE = 11000000;
	unsigned int numParticles;
	float4* particles;
	float4* newParticles;
	unsigned int* tree;
	unsigned int* treeSize;
	unsigned int* particlesDone;
	LBABP lbws;
	LBStatic lbstat;
	
	float totalTime;
	LBMethod method;
public:
	void generateParticles();
	bool run(unsigned int threads, unsigned int blocks, LBMethod method, int maxChildren, int numParticles);
	float printStats();
	float getTime();
	int getMaxMem();
};

