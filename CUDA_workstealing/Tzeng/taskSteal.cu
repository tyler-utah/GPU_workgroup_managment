/*
taskSteal routine goes here
*/

__global__ void taskSteal(float * d_inputData, float * d_outputData, float * d_debugData, int * d_intDebugData,
						   int * d_locks, int * d_headQueuePtr, int * d_tailQueuePtr,
						   float * d_rands, int numElements, int *d_outQueuePtr,
						   int * d_numPatches, int startHeadPtr, SCHEDULE_RESULTS results)	//tail defaults to start at 0
{
	int tid = threadIdx.x;
    int bid = blockIdx.x;

	__shared__ volatile float inputData[128]; // only one patch per block
    __shared__ volatile float outputData[256]; // but we may store two patches


    __shared__ volatile int tryBlock;
    __shared__ volatile int stoleWork;	//1 if we did steal work
	__shared__ volatile int gotWork;
	__shared__ volatile int hasWork;
	__shared__ volatile int writeBackToTail;

    __shared__ volatile int currQueueLoc;
    __shared__ volatile int endQueueLoc;
    __shared__ volatile int auxQueueLoc;
	__shared__ volatile int headQueueLoc;
	__shared__ volatile int outQueueLoc;
    __shared__ volatile	int numPatches;
	__shared__ volatile int globalPatches;

	__shared__ volatile int stillAlive;		//the cake is a lie!
	__shared__ volatile int numPatchesWriteBack;
	__shared__ volatile float dumbptr;
	__shared__ volatile int baseLoc;


	//data to be written out
	__shared__ volatile int numIterations;
	__shared__ volatile int numLocksUsed;
	__shared__ volatile int maxMemoryUsed;	//can be measured as the maximum amount of patches left
	__shared__ volatile int numPatchesFlushedOut;
	__shared__ volatile int numIdle;
	__shared__ volatile int numSteals;
	__shared__ volatile int maxPatches;		//the maximum number of patches ever in this dequeue
	__shared__ volatile int numPatchesTaken;

	//initialize all shared memory
	if(tid ==0)
	{
		tryBlock = bid;
		stoleWork = 0;
		gotWork = 0;
		writeBackToTail = 0;
		globalPatches = numElements;
		numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
		stillAlive = 1;
		numIterations = 0;
		baseLoc = bid * TS_BUCKET_SIZE;
		hasWork = 0;
		numLocksUsed =0;
		numPatchesFlushedOut=0;
		numPatchesWriteBack=0;
		numIdle = 0;
		numSteals = 0;
		maxPatches = numPatches;
		numPatchesTaken=0;
		auxQueueLoc=0;
	}//end if

	while(1)
	{		
	//	if(tid==0 && numIterations < TS_MAX_ITERATIONS)
	//		results.d_workInQueue[bid*TS_MAX_ITERATIONS+numIterations] = numPatches;

		numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
		//update the maximum number of patches in this bin
		if(tid==0)
		{
			maxPatches = (numPatches > maxPatches) ? numPatches : maxPatches;
			numIterations++;
		}
		if(globalPatches <= 0 && numPatches <= 0)
		{
			if(tid == 0)
			{
				//atomicAdd(g_numPatchesLeft, 16);
				d_locks[bid] = 0;	//just as a safety precaution
				//if(tryBlock != -1) d_locks[tryBlock]=0;
			}
			stillAlive = 0;
			break;
		}//end break condition

	//	if(numPatchesTaken >= 800) break;
		
		//fetch work
		if(tid ==0 && hasWork == 0)
		{
			if(numPatches < 1*64)
			{
				if(globalPatches > 64)
				{
					//steal
					numSteals++;
					tryBlock = taskSteal(currQueueLoc, d_headQueuePtr, d_tailQueuePtr, d_locks);
					stoleWork = 1;
					gotWork = (tryBlock == -1) ? 0 : 1;
					numLocksUsed += (tryBlock == -1) ? 0 : 1;
					numPatchesTaken++;
				}
				else
				{
					//idle for one turn
					//we can't do anything
					tryBlock = -1;
					stoleWork =0;
					gotWork = 0;
				}
			}//end if numPatches
			else if(numPatches == 1*64)
			{
				//read the patch from the head
				numLocksUsed++;
				while(atomicCAS(&d_locks[bid], 0, 1) ==1);
                //think of this as stealing work from ourselves
                //since there is one block left, take it from the head
                stoleWork = 1;
                tryBlock = bid;
                //currQueueLoc = subtractFromHead(bid, 64, d_headQueuePtr);
				gotWork = 1;
				numPatchesTaken++;
			}
			else
			{
				//we still got our own work to do...
			  //currQueueLoc = appendToTail(bid, 64, d_tailQueuePtr);
				tryBlock = bid;
				stoleWork = 0;
				gotWork = 1;
				numPatchesTaken++;
			}//end else
		}//end if tid == 0

		
		//if we failed to get work...
		if (hasWork == 0 && gotWork == 0)
        {
			//update values here
			if(tid==0)
			{
				globalPatches = *d_numPatches;
				numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
				//add idle counter here
				numIdle++;
			}
            continue;	//tried to steal work but nothing, try again
        }

		if(hasWork == 0)
		{
			//read in the data, there should be data to read now
	        inputData[tid +  0] = d_inputData[currQueueLoc +  0 + tid];
	        inputData[tid + 32] = d_inputData[currQueueLoc + 32 + tid];
		
			if(tid == 0 && stoleWork)
			{
				//now we restore the lock
				d_locks[tryBlock] = 0; 
			}

			if(tid == 0){globalPatches = atomicSub(d_numPatches, 64); globalPatches-=64;}
		}//end hasWork ==0

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
			}

			d_outputData[outQueueLoc + tid +0 ] = outputData[tid +0 ];
			d_outputData[outQueueLoc + tid +32] = outputData[tid +32];

			globalPatches = *d_numPatches;
//			if(numIterations == 1) break;
			continue;
		}//end if

		//here we need to write out the work back into the queue
		//important step: check first if we can write everything out to the tail!
		//estimate the usage in the tail
		if(tid==0)
		{
			currQueueLoc = d_tailQueuePtr[bid];
			if(currQueueLoc >= (numPatchesWriteBack-1)*64)
				writeBackToTail = 1;
			else
				writeBackToTail = 0;
		}

		if(writeBackToTail)
		{
			//we can save a lock and write it back to the tail!
			//use currQueueLoc to calculate the address
			for(int i=0; i<numPatchesWriteBack-1;i++)
			{
				
				if(tid == 0) auxQueueLoc = currQueueLoc + baseLoc - (64* (i+1));

				d_inputData[auxQueueLoc + tid +0 ] = inputData[tid+0 ];
				d_inputData[auxQueueLoc + tid +32] = inputData[tid+32];
				
			}//end for			

			//once all the patches are written back, update the tail
			if(tid == 0)
			{
				subtractFromTail(bid, (numPatchesWriteBack-1)*64, d_tailQueuePtr);
				globalPatches = atomicAdd(d_numPatches, 64*(numPatchesWriteBack-1));
				globalPatches += 64*(numPatchesWriteBack-1);
				numPatchesTaken++;
			}
//			if(numIterations == 1) break;
			continue;
		}
//if(numIterations == 1) break;
		//need to write back to head
		//grab the lock
		if(tid ==0)
		{
			numLocksUsed++;
			while(atomicCAS(&d_locks[bid], 0,1) == 1);
		}
		for(int i=0; i<numPatchesWriteBack-1; i++)
		{
			if(tid == 0)
				headQueueLoc = atomicAdd(&d_headQueuePtr[bid], 64);

			d_inputData[headQueueLoc + tid +0 ] = inputData[tid+0 ];
			d_inputData[headQueueLoc + tid +32] = inputData[tid+32];
		}//end for

		//now release the lock
		if(tid==0)
		{
			globalPatches = atomicAdd(d_numPatches, 64*(numPatchesWriteBack-1));
			globalPatches += 64*(numPatchesWriteBack-1);
			d_locks[bid] = 0;
			numPatchesTaken++;
		}

		
	}//end while(1)
		//at the end, write out data into the debug array for reference
	if(tid == 0)
	{
		results.d_numIterations[bid] = numIterations;
		results.d_numLocks[bid] = numLocksUsed;
		results.d_maxMemUsage[bid] = maxPatches;
		results.d_numIdle[bid] = numIdle;
		results.d_numSteals[bid] = numSteals;
		results.d_numPatchesTaken[bid] = numPatchesTaken;
		results.d_numPatchesProcessed[bid] = numPatchesFlushedOut;

		d_debugData[bid*7 +0] = d_tailQueuePtr[bid];
		d_debugData[bid*7 +1] = currQueueLoc;
		d_debugData[bid*7 +2] = d_headQueuePtr[bid];
		d_debugData[bid*7 +3] = numPatchesWriteBack;
		d_debugData[bid*7 +4] = numPatchesFlushedOut;
		d_debugData[bid*7 +5] = numPatches;
		d_debugData[bid*7 +6] = auxQueueLoc;
	}
}//end tasksteal
