/*
block Queue kernel
*/
//Here is the convention:
//Read from TAIL
//Write to HEAD

__global__ void blockQueue(float * d_inputData, float * d_outputData, float * d_debugData, int * d_intDebugData,
						   int * d_locks, int * d_headQueuePtr, int * d_tailQueuePtr,
						   float * d_rands, int numElements, int *d_outQueuePtr,
						   int * d_numPatches, int startHeadPtr, SCHEDULE_RESULTS results)	//tail defaults to start at 0
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;

	//perhaps we can change this later to let block take bigger chunks of data at once
	__shared__ volatile float inputData[256];		//this allows us to read in 4 blocks of work at once
	__shared__ volatile float outputData[64];		//we write out 1 block at a time

	//other shared values
	__shared__ volatile int tailQueueLoc;
	__shared__ volatile int headQueueLoc;
	__shared__ volatile int outQueueLoc;
	__shared__ volatile int hasWork;
	__shared__ volatile int globalPatchesLeft;
	__shared__ volatile int fetchedFromHead;

	__shared__ volatile int numPatchesWriteBack;
	__shared__ volatile float dumbptr;
	__shared__ volatile int numPatchesFlushedOut;
	__shared__ volatile int numPatchesTaken;

	//data to be written out
	__shared__ volatile int numLocksUsed;
	__shared__ volatile int maxMemoryUsed;	//can be measured as the maximum amount of patches left

	__shared__ volatile int idleTime;

	int numIterations = 0;
	//initialize everything
	if(tid == 0)
	{
		tailQueueLoc = 0;
		headQueueLoc = startHeadPtr;
		outQueueLoc = 0;
		hasWork = 0;
		globalPatchesLeft = numElements;
		numPatchesWriteBack = 0;
		numLocksUsed = 0;
		numPatchesFlushedOut=0;
		idleTime = 0;
		numPatchesTaken =0;
	}
	numIterations = 0;
	
	//start the blockQueue kernel
	while(1)
	{
		numIterations++;	
		//exit condition
		
		if(numIterations >= 300) break;
//		if(numPatchesTaken >= 300) break;
		globalPatchesLeft = *d_numPatches;

		if(globalPatchesLeft <= 0) break;

		if(tid == 0 && hasWork == 0)
		{
			//we need to fetch work
			//check the number of patches left
			if(globalPatchesLeft < 5 * 64)
			{
				//lock it and fetch from the head
				numLocksUsed++;
				while(atomicCAS(&d_locks[HEAD_LOCK], 0,1) == 1) idleTime++;
				//tailQueueLoc here is really a misnomer
				tailQueueLoc = atomicSub(d_headQueuePtr, 64);
				tailQueueLoc -= 64;
				fetchedFromHead =1;
			}
			else
			{
				//fetch from the tail
				tailQueueLoc = atomicAdd(d_tailQueuePtr, 64);
				fetchedFromHead =0;
				
			}
			numPatchesTaken++;
		}//end if tid==0

		//now fetch work
		if(hasWork == 0)
		{
			inputData[tid +  0] = d_inputData[tailQueueLoc +  0 + tid];
			inputData[tid + 32] = d_inputData[tailQueueLoc + 32 + tid];

			if(tid == 0)
			{
				hasWork =1;
				globalPatchesLeft = atomicSub(d_numPatches, 64);
				globalPatchesLeft -= 64;

				if(fetchedFromHead == 1)
					d_locks[HEAD_LOCK] = 0;
			}
		}//end fetch work

		//now we pretend to process the work
		//threadID 0 does the work
		if(tid == 0)
		{
			//pick something out of the rand memory 
			//we must make sure later on that this never gets out of memory
			int rand_loc = tid + bid * blockDim.x + numIterations;
			rand_loc %= NUM_RANDS;
			float randNum = d_rands[rand_loc];
			numPatchesWriteBack = doSynthWork(&dumbptr, randNum);
		}

		
		//now we need to write back, or fetch more work
		if(tid == 0) hasWork = (numPatchesWriteBack == NO_MORE_WORK) ? 0 : 1;

		if(numPatchesWriteBack == NO_MORE_WORK)
		{
			outputData[tid] = dumbptr;
			outputData[tid+32] = dumbptr;

			//we need to write something out here
			if(tid==0)
			{
				outQueueLoc = atomicAdd(d_outQueuePtr,64);
				numPatchesFlushedOut++;
				hasWork = 0;
				d_locks[HEAD_LOCK] = 0;
			}
			d_outputData[outQueueLoc + tid +0 ] = outputData[tid +0 ];
			d_outputData[outQueueLoc + tid +32] = outputData[tid +32];
			
			continue;
		}//end if

		//if we are going to write it back, we can add extra information to that single value here
		//do it here

		//here we need to write out the work back into the queue
		//grab the lock
		if(tid ==0)
		{
			numLocksUsed++;
			while(atomicCAS(&d_locks[HEAD_LOCK], 0,1) == 1) idleTime++;
		}

		for(int i=0; i<numPatchesWriteBack-1; i++)
		{
			if(tid == 0)
			{
				headQueueLoc = atomicAdd(d_headQueuePtr, 64);
				globalPatchesLeft = atomicAdd(d_numPatches, 64);
			}

			d_inputData[headQueueLoc + tid +0 ] = inputData[tid+0 ];
			d_inputData[headQueueLoc + tid +32] = inputData[tid+32];
		}//end for

		//now release the lock
		if(tid==0)
		{
//			globalPatchesLeft = atomicAdd(d_numPatches, 64*(numPatchesWriteBack-1));
//			globalPatchesLeft += 64*(numPatchesWriteBack-1);
			d_locks[HEAD_LOCK] = 0;

			numPatchesTaken++;	//holding onto one patch = one more unit of work for us
		}

	}//end while(1)

	//at the end, write out data into the debug array for reference
	if(tid == 0)
	{
		results.d_numIterations[bid] = numIterations;
		results.d_numLocks[bid] = numLocksUsed;
		results.d_numIdle[bid] = idleTime;
		results.d_numPatchesTaken[bid] = numPatchesTaken;
		results.d_numPatchesProcessed[bid] = numPatchesFlushedOut;

		d_debugData[bid*6 +0] = tailQueueLoc;
		d_debugData[bid*6 +1] = globalPatchesLeft;
		d_debugData[bid*6 +2] = headQueueLoc;
		d_debugData[bid*6 +3] = numPatchesWriteBack;
		d_debugData[bid*6 +4] = idleTime;
		d_debugData[bid*6 +5] = numPatchesFlushedOut;
	}
}//end blockQueue
