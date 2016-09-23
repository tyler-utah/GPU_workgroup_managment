/*
  taskDonate routine goes here
*/


__global__ void taskDonate(float * d_inputData, float * d_outputData, float * d_debugData, int * d_intDebugData,
			   int * d_locks, int * d_headQueuePtr, int * d_tailQueuePtr,
			   float * d_rands, int numElements, int *d_outQueuePtr,
			   int * d_numPatches, int startHeadPtr, SCHEDULE_RESULTS results)	//tail defaults to start at 0
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
 
  __shared__ volatile float inputData[128]; // only one patch per block
  __shared__ volatile float outputData[256]; // but we may store two patches

  __shared__ volatile int bail;
  __shared__ volatile int tryBlock;
  __shared__ volatile int stoleWork;	//1 if we did steal work
  __shared__ volatile int gotWork;
  __shared__ volatile int hasWork;
  __shared__ volatile int writeBackToTail;
  __shared__ volatile int doWeDonate;

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
  __shared__ volatile int numDonates;
  __shared__ volatile int maxPatches;		//the maximum number of patches ever in this dequeue
  __shared__ volatile int numPanics;
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
      numDonates= 0;
      maxPatches = numPatches;
      doWeDonate = 0;
      currQueueLoc = 0;
      numPanics = 0;
      numPatchesTaken=0;
      bail = 0;
    }//end if

  while(1)
    {		
      //update the maximum number of patches in this bin	
      //take a look at this PTX....not sure why this causes crashes....
      numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
      if(tid ==0)
	{
	  maxPatches = (numPatches > maxPatches) ? numPatches : maxPatches;
	  numIterations++;
	}

      if(globalPatches <= 0 && numPatches <= 0)
	{
	  // if(tid == 0)
	  //   {		
	  // 	d_locks[bid] = 0;	//just as a safety precaution
	  //   }
	  stillAlive = 0;
	  break;
	}//end break condition

      //if(numPatchesTaken >= 800) break;
		
      //fetch work
      if(tid ==0 && hasWork == 0) {
	if(numPatches < 1*64) {
	  if(globalPatches >= 10*64){
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
	    lock(&d_locks[bid]);
	    //while(atomicCAS(&d_locks[bid], 0, 1) ==1);

	    numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
	    if (numPatches < 64) {
	      unlock(&(d_locks[bid]));
	      //d_locks[bid] = 0;
	      tryBlock = -1;
	      stoleWork = 0;
	      gotWork = 0;
	      goto failed;
	    }

	    //think of this as stealing work from ourselves
	    //since there is one block left, take it from the head
	    gotWork = 1;
	    numPatchesTaken++;
	    stoleWork = 1;
	    tryBlock = bid;
	    subtractFromHead(bid, 64, d_headQueuePtr);
	  }
	else
	  {
	    //we still got our own work to do...
	    lock(&d_locks[bid]);
	    //while(atomicCAS(&d_locks[bid], 0, 1) ==1);

	    numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
	    if (numPatches < 64) {
	      unlock(&(d_locks[bid]));
	      //d_locks[bid] = 0;
	      tryBlock = -1;
	      stoleWork = 0;
	      gotWork = 0;
	      goto failed;
	    }

	    appendToTail(bid, 64, d_tailQueuePtr);
	    unlock(&(d_locks[bid]));
	    //d_locks[bid] = 0;
	    tryBlock = bid;
	    stoleWork = 0;
	    gotWork = 1;
	    numPatchesTaken++;
	  }//end else
      }//end if tid == 0

      //if we failed to get work...
    failed:
      if (tryBlock == -1) {
	if(tid==0) {
	  //update values here
	  globalPatches = *d_numPatches;
	  numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
	  //add idle counter here
	  numIdle++;
	  tryBlock = bid;
	}
	continue;	//tried to steal work but nothing, try again
      }

      if(hasWork == 0) {
	//read in the data, there should be data to read now
	if(tid == 0 && stoleWork) {
	  //now we restore the lock		    
	  unlock(&(d_locks[tryBlock]));
	  //d_locks[tryBlock] = 0; 
	}

	if(tid == 0){
	  globalPatches = atomicSub(d_numPatches, 64); 
	  globalPatches-=64;
	}
      }//end hasWork ==0
				
		
      //now we pretend to process the work
      //threadID 0 does the work
      if(tid == 0)
	{
	  //pick something out of the rand memory 
	  //we must make sure later on that this never gets out of memory
	  //int rand_loc = tid + bid * blockDim.x + numIterations;
	  int rand_loc = atomicInc(&rand_index, NUM_RANDS);
	  rand_loc %= NUM_RANDS;
	  float randNum = d_rands[rand_loc];
	  numPatchesWriteBack = doSynthWork(&dumbptr, randNum);
	}

      //now we need to write back, or fetch more work
      if(tid == 0) hasWork = (numPatchesWriteBack == NO_MORE_WORK) ? 0 : 1;

	
      if(numPatchesWriteBack == NO_MORE_WORK) {
	// outputData[tid] = dumbptr;
							
	// outputData[tid+32] = dumbptr;

	//we need to write something out here
	if(tid==0) {
	  //outQueueLoc = atomicAdd(d_outQueuePtr,64);
	  numPatchesFlushedOut++;
	  globalPatches = *d_numPatches;
	  numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
	}

	//d_outputData[outQueueLoc + tid +0 ] = outputData[tid +0 ];
	//d_outputData[outQueueLoc + tid +32] = outputData[tid +32];
			
	continue;
      }//end if
	
      //here we need to write out the work back into the queue
      //important step: check first if we can write everything out to the tail!
      //estimate the usage in the tail
      if(tid==0) {
	currQueueLoc = d_tailQueuePtr[bid];
	if(currQueueLoc >= (numPatchesWriteBack-1)*64)
	  writeBackToTail = 1;
	else
	  writeBackToTail = 0;
      }
	
      if(writeBackToTail) {
	//we can save a lock and write it back to the tail!
	//use currQueueLoc to calculate the address
	// for(int i=0; i<numPatchesWriteBack-1;i++) {
				
	// 	if(tid == 0) auxQueueLoc = currQueueLoc + baseLoc - (64* (i+1));

	// 	// d_inputData[auxQueueLoc + tid +0 ] = inputData[tid+0 ];
								
	// 	// d_inputData[auxQueueLoc + tid +32] = inputData[tid+32];
				
	//   }//end for

	if (tid == 0) {
	  numLocksUsed++;
	  lock(&d_locks[bid]);
	  //while(atomicCAS(&d_locks[bid], 0,1) == 1); 
	  currQueueLoc = d_tailQueuePtr[bid];
	  if(currQueueLoc >= (numPatchesWriteBack-1)*64)
	    bail = 0;
	  else
	    bail = 1;
	}
	if (bail) {
	  unlock(&(d_locks[bid]));
	  //d_locks[bid] = 0;
	  goto not_tail;
	}

	  
	//once all the patches are written back, update the tail
	if(tid == 0)
	  {
	    subtractFromTail(bid, (numPatchesWriteBack-1)*64, d_tailQueuePtr);
	    unlock(&(d_locks[bid]));
	    //d_locks[bid] = 0;
	    globalPatches = atomicAdd(d_numPatches, 64*(numPatchesWriteBack-1));
	      
	    globalPatches += 64*(numPatchesWriteBack-1);
	    //			globalPatches = *d_numPatches;
	    numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
	    numPatchesTaken++;
	  }
	continue;
      }
    not_tail:

      //-----------------------
      //insert donation check here
      //-----------------------
      //check first to see if we need to donate
      // if(tid==0)  {
      //     headQueueLoc = d_headQueuePtr[bid];
      //     if(headQueueLoc + (numPatchesWriteBack-1)*64 >= 1200*64)
      //       doWeDonate = 1;
      //     else
      //       doWeDonate = 0;
      //   }//end if tid

      // assert(!doWeDonate);

      //final case: write out to the head of your own queue
      //need to write back to head
      //grab the lock
      if(tid ==0) {
	numLocksUsed++;
	lock(&d_locks[bid]);
	//while(atomicCAS(&d_locks[bid], 0,1) == 1);			
      }
      for(int i=0; i<numPatchesWriteBack-1; i++) {
	if(tid == 0) {
	  //headQueueLoc = atomicAdd(&d_headQueuePtr[bid], 64);
	  //d_headQueuePtr[bid]+=64;
	  appendToHead(bid,64,d_headQueuePtr);
	}
	    
	//d_inputData[headQueueLoc + tid +0 ] = inputData[tid+0 ];
	//d_inputData[headQueueLoc + tid +32] = inputData[tid+32];
			  
      }//end for

      //now release the lock
      if(tid==0) {

	//This doesn't need to be in the critical
	//section. I'll move it after to get the important
	//accesses closer together - Tyler
	//globalPatches = atomicAdd(d_numPatches, 64*(numPatchesWriteBack-1));
	//globalPatches += 64*(numPatchesWriteBack-1);
		  
	// d_headQueuePtr[bid] += 64;
	//atomicAdd(&d_headQueuePtr[bid], 64 * (numPatchesWriteBack-1));
	//d_headQueuePtr[bid] += 64 * (numPatchesWriteBack-1);
	unlock(&(d_locks[bid]));
	//d_locks[bid] = 0;


	globalPatches = atomicAdd(d_numPatches, 64*(numPatchesWriteBack-1));		  
	globalPatches += 64*(numPatchesWriteBack-1);

	numPatchesTaken++;
	//			globalPatches = *d_numPatches;
	numPatches = getNumPatches(bid, d_headQueuePtr, d_tailQueuePtr);
      }

    }//end while(1)
  //at the end, write out data into the debug array for reference
  if(tid == 0) {
    write_reference_data(&results.d_numIterations[bid], numIterations);
    //results.d_numIterations[bid] = numIterations;
    write_reference_data(&results.d_numLocks[bid], numLocksUsed);
    //results.d_numLocks[bid] = numLocksUsed;
    write_reference_data(&results.d_maxMemUsage[bid], maxPatches);
    //results.d_maxMemUsage[bid] = maxPatches;
    write_reference_data(&results.d_numIdle[bid], numIdle);
    //results.d_numIdle[bid] = numIdle;
    write_reference_data(&results.d_numSteals[bid], numSteals);
    //results.d_numSteals[bid] = numSteals;     
    //results.d_numDonations[bid] = numDonates;
    write_reference_data(&results.d_errorMsg[bid], numPanics);
    //results.d_errorMsg[bid] = numPanics;
    write_reference_data(&results.d_numPatchesTaken[bid], numPatchesTaken);
    //results.d_numPatchesTaken[bid] = numPatchesTaken;
    write_reference_data(&results.d_numPatchesProcessed[bid], numPatchesFlushedOut);
    //results.d_numPatchesProcessed[bid] = numPatchesFlushedOut;

    write_reference_data(&d_debugData[bid*7 +0], d_tailQueuePtr[bid]);
    //d_debugData[bid*7 +0] = d_tailQueuePtr[bid];
    write_reference_data(&d_debugData[bid*7 +1], currQueueLoc);
    //d_debugData[bid*7 +1] = currQueueLoc;
    write_reference_data(&d_debugData[bid*7 +2], numPatchesWriteBack);
    //d_debugData[bid*7 +2] = numPatchesWriteBack;
    write_reference_data(&d_debugData[bid*7 +3], tryBlock);
    //d_debugData[bid*7 +3] = tryBlock;
    write_reference_data(&d_debugData[bid*7 +4], numPatchesFlushedOut);
    //d_debugData[bid*7 +4] = numPatchesFlushedOut;
    write_reference_data(&d_debugData[bid*7 +5], writeBackToTail);
    //d_debugData[bid*7 +5] = writeBackToTail;
    write_reference_data(&d_debugData[bid*7 +6], hasWork);
    //d_debugData[bid*7 +6] = hasWork;
  }
}//end taskdonate
