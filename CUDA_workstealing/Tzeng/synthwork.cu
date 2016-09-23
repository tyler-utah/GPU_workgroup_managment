#include "synthwork.h"
#include "globals.h"
#include <cuda_runtime_api.h>
#include "helper_timer.h"
#include <stdlib.h>

#define CUT_SAFE_CALL 
#define CUT_CHECK_ERROR


#include "cudaHelperFunc.cu"

// =========================
//      Constant Memory
// =========================

// work = {q,m, worksize_when_producing, worksize_when_consuming}
__device__ float c_workdesc[4]; // work descriptor

__device__ unsigned int rand_index;

__global__ void init_c_workdesc(float f1, float f2, float f3, float f4) {
  c_workdesc[0] = f1;
  c_workdesc[1] = f2;  
  c_workdesc[2] = f3;
  c_workdesc[3] = f4;
  rand_index = 0;
  
  return;
}

// =========================
//      GPU Functions
// =========================


// this function occupies the GPU for 
// a while

__device__ inline float occupyGPU(volatile float* dumbptr,  //single float
								 int nops){
    // this function occupies the GPU for n ops

    float acc = *dumbptr;
    
    float val1 = 1.001;
    float val2 = 0.999;

#pragma unroll
    for(int i=0; i<nops; i++){
        acc+=acc*val1;
    }
#pragma unroll
    for(int i=0; i<nops; i++){
        acc+=acc*val2;
    }
    *dumbptr = acc;
	return acc;
}

// this function either does some work or creates new items

__device__ inline int doSynthWork(volatile float *dumbptr, float unifrand  //randbuffer[objectID]
								){

  if(unifrand < c_workdesc[0]){
    occupyGPU(dumbptr, c_workdesc[2]);
    int ret = c_workdesc[1];
    return ret;
  }else{
    occupyGPU(dumbptr, c_workdesc[3]);
    // full consumption
    return NO_MORE_WORK;
  }
  
}

//-----------------------------------------------------------
//These device functions are for the task steal and task donate 
//routines
//Appends to the head pointer, this is needed to deal with wrap around cases
__device__ __forceinline__ void appendToHead(int bid, int value, int * d_dequeHeadPtr)
{
  //int binBase = bid * TS_BUCKET_SIZE;
  //int currHead = d_dequeHeadPtr[bid];
  
  d_dequeHeadPtr[bid] += value;
  
  
  //currHead += value;
  
  //d_dequeHeadPtr[bid] = currHead;
  

  //int returnLoc = currHead % TS_BUCKET_SIZE - value + binBase;
  //return returnLoc;	//here we want the value before the addition
}

//Appends to the tail pointer, this is needed to deal with wrap around cases
__device__ __forceinline__ void appendToTail(int bid, int value, int * d_dequeTailPtr)
{
  //int binBase = bid * TS_BUCKET_SIZE;
    // int currTail = d_dequeTailPtr[bid];
    
    // currTail += value;

    // d_dequeTailPtr[bid] = currTail;
  d_dequeTailPtr[bid] += value;
    

    //int returnLoc = currTail % TS_BUCKET_SIZE - value + binBase;
    //return -1;		//essentailly you want the value before the addition
}

//Subtract from the head poitner
__device__ inline void subtractFromHead(int bid, int value,int * d_dequeHeadPtr)
{
  //int binBase;
  //based on the blockID, compute the bucket boundaries

    
    d_dequeHeadPtr[bid] -= value;
    
    //binBase = TS_BUCKET_SIZE * bid;
    //return currHead % TS_BUCKET_SIZE + binBase;	//this becomes the currLoc
}

//Subtract from the tail poitner
__device__ __forceinline__ void subtractFromTail(int bid, int value,int * d_dequeTailPtr)
{
  //int binBase;
  //based on the blockID, compute the bucket boundaries
  
  //binBase = TS_BUCKET_SIZE * bid;
  d_dequeTailPtr[bid] -= value;
  
  // currTail -= value;
  
  // d_dequeTailPtr[bid] = currTail;
  
  // return currTail % TS_BUCKET_SIZE + binBase;	//this becomes the currLoc
}

__device__ __forceinline__ void lock(int* mutex) {
  while(atomicCAS(mutex, 0, 1) ==1);
}

__device__ __forceinline__ void unlock(int* mutex) {
  *mutex = 0;
}

__device__ __forceinline__ void write_reference_data(int* addr, const int data) {
  *addr = data;
}

__device__ __forceinline__ void write_reference_data(float* addr, const float data) {
  *addr = data;
}


//this returns the number of patches left in a block
__device__ __forceinline__ int getNumPatches(int bid, int * d_dequeHeadPtr, int * d_dequeTailPtr) {
    int currHead, currTail;

    currHead = d_dequeHeadPtr[bid];    
    currTail = d_dequeTailPtr[bid];    

    // if(currHead == currTail)
    //   return 0;

    // int numPatches = 0;    
    // numPatches = currHead - currTail;
    
    return currHead - currTail;	//for synth workloads we do not divide by 64 and use the actual value
}

__device__ int taskSteal(volatile int & currQueueLoc, int * d_dequeHeadPtr, int * d_dequeTailPtr, int * d_locks)
{
    int tryBlock = blockIdx.x;
    //keep trying to query a new block's patches
    do
    {
        tryBlock = (tryBlock+1) % gridDim.x;

        int numPatches = getNumPatches(tryBlock, d_dequeHeadPtr, d_dequeTailPtr);
        
	if( numPatches > 128) {
            //this guy has more than 2 patches, we can grab one!
            //get lock
	  if(atomicCAS(&d_locks[tryBlock], 0,1)==1) continue;
	  
            //we have the lock, now grab data from the head
	  //currQueueLoc = subtractFromHead(tryBlock, 64, d_dequeHeadPtr);
	  
	  //Tyler: Fixing the bug, while trying to create a nice situation for
	  //weak memory to cause problems :)

	  //Tyler: Check to see if there is *still* enough data to steal.
	  numPatches = getNumPatches(tryBlock, d_dequeHeadPtr, d_dequeTailPtr);
	  // int currHead = d_dequeHeadPtr[tryBlock];
	  // int currTail = d_dequeTailPtr[tryBlock];
	  // numPatches = currHead - currTail;

	  //Tyler: If not, exit.
	  if (numPatches < 64) {
	    unlock(&d_locks[tryBlock]);
	    //d_locks[tryBlock] = 0;
	    continue;
	  }
	    
	  //Tyler: essentially just inline subtractFromHead here
	  subtractFromHead(tryBlock,64,d_dequeHeadPtr);
	  // currHead -= 64;
	  // d_dequeHeadPtr[tryBlock] = currHead;
	  // int binBase = TS_BUCKET_SIZE * tryBlock;
	  
	  // currQueueLoc = currHead % TS_BUCKET_SIZE + binBase;
	  
	  //note that at this point in time WE HAVEN'T RELEASED THE LOCK
	  //we release it when everything has been read (i.e. in the main kernel code
            return tryBlock;	//return the block from which we are stealing
        }
    }while(tryBlock != blockIdx.x);//end while

	return -1;
}//end worksteal



//-----------------------------------------------------------
//Kernels
#include "blockQueue.cu"
#include "taskSteal.cu"
#include "taskDonate.cu"

// =========================
//      CPU Functions
// =========================

// performs any initialization needed

void initSynthWork( float exit_probability, 
                    int nchildren, 
                    int worksize_when_producing, 
                    int worksize_when_consuming,
                    float *randbuffer,
                    int nrands
                    ){

    // float workdescriptor[4] = {exit_probability, 
    //                             (float)nchildren, 
    //                             (float)worksize_when_producing, 
    //                             (float)worksize_when_consuming};

    // cutilSafeCall(cudaMalloc((void**)&c_workdesc, sizeof(float)*4));
    // cutilSafeCall(cudaMemcpyToSymbol(c_workdesc,    workdescriptor,     4*sizeof(float),0,cudaMemcpyHostToDevice));
    init_c_workdesc<<<1,1>>>(exit_probability, (float)nchildren,  (float)worksize_when_producing, (float)worksize_when_consuming);

    float *hrands = new float[nrands];

    srand(0);
    for(int i=0; i<nrands; i++){
      hrands[i] = (float)rand()/(float)RAND_MAX;
    }
    srand(time(NULL));

	cudaMemcpy(randbuffer, hrands, sizeof(float) * nrands, cudaMemcpyHostToDevice);

    delete [] hrands;
    
}

//[stan] anjul I am confused, why do we need all these params to free the synth work?
void deleteSynthWork( float exit_probability, 
                    int nchildren, 
                    int worksize_when_producing, 
                    int worksize_when_consuming,
                    float *randbuffer,
                    int nrands
                    ){

cudaFree(randbuffer);
}

float my_fabs(float a) {
  if (a < 0) 
    return -a ;
  else 
    return a;
}

//-----------------------------------------------------------
//Launchers
float Launch_Synth_Scheduler(float * d_inputData, float * d_outputData, float * d_debugData,
							int * d_intDebugData, int * d_headQueuePtr, int * d_tailQueuePtr,
							int * d_locks, float * d_randData, int numElements, SCHEDULE_RESULTS results)
{
	dim3 grd(TS_NUM_BUCKETS,1,1), blk(32,1,1);

	//a location pointer to the output Queue
	int * d_outputQueueLoc;
	cutilSafeCall(cudaMalloc((void**)&d_outputQueueLoc, sizeof(int)));
	cutilSafeCall(cudaMemset(d_outputQueueLoc, 0, sizeof(int)));

	//another one for the number of patches
	int * d_numPatches;
	cutilSafeCall(cudaMalloc((void**)&d_numPatches, sizeof(int)));
	cutilSafeCall(cudaMemset(d_numPatches, 0, sizeof(int)));
	cutilSafeCall(cudaMemcpy(d_numPatches, &numElements, sizeof(int), cudaMemcpyHostToDevice));

	//try 1 block scenario first	
	//set up the data here
	cutilSafeCall(cudaMemset(d_debugData, 0, sizeof(float)*MAX_ARR));
	
	if(scheduler == RUN_BLOCK) {
	  cutilSafeCall(cudaMemset(d_locks, 0, sizeof(int)*TS_NUM_BUCKETS));
	}
	else {
	  cutilSafeCall(cudaMemset(d_locks, 0, sizeof(int)));
	}

	StopWatchInterface * timer;
	CUT_SAFE_CALL(sdkCreateTimer(&timer));
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	//CUT_CHECK_ERROR("Before scheduler");
	//now launch the kernels
	if(scheduler == RUN_TASKSTEAL)
	{
		printf("running task steal...\n");
		assert(0);

		taskSteal<<<grd,blk>>>(d_inputData, d_outputData, d_debugData, d_intDebugData, d_locks, d_headQueuePtr, d_tailQueuePtr, d_randData, numElements, d_outputQueueLoc,
							d_numPatches, numElements, results);
	}
	else if (scheduler == RUN_TASKDONATE)
	{
	  printf("running task donation...\n");
	  
	  taskDonate<<<grd,blk>>>(d_inputData, d_outputData, d_debugData, d_intDebugData, d_locks, d_headQueuePtr, d_tailQueuePtr, d_randData, numElements, d_outputQueueLoc,
						 d_numPatches, numElements, results);
	}
	else
	  {
		printf("running block queue...\n");
		//block queue
		blockQueue<<<grd,blk>>>(d_inputData, d_outputData, d_debugData, d_intDebugData, d_locks, d_headQueuePtr, d_tailQueuePtr, d_randData, numElements, d_outputQueueLoc,
							d_numPatches, numElements, results);
	}


	//CUT_CHECK_ERROR("After scheduler");

	sdkStopTimer(&timer);
	float time = sdkGetTimerValue(&timer);

	if(scheduler == RUN_BLOCK)
	{
// 	  debugPrintScreen(results.d_numIterations, TS_NUM_BUCKETS, const_cast<char*>("iterations"), 1);
// 		debugPrintScreen(results.d_numLocks, TS_NUM_BUCKETS, "numLocks", 1);
// 		debugPrintScreen(results.d_numPatchesTaken, TS_NUM_BUCKETS, "num patches taken", 1);
// 		debugPrintScreen(results.d_numIdle, TS_NUM_BUCKETS, "idle Time", 1);

// //		debugPrintScreen(d_debugData, 6*TS_NUM_BUCKETS, "debug Info", 6);
// 		debugPrintScreen(d_numPatches, 1, "global numPatches", 1);
// 		debugPrintScreen(d_outputQueueLoc, 1, "outputQueueSize", 1);
	}
	else if(scheduler == RUN_TASKDONATE)
	{
	  //	  debugPrintScreen(results.d_numIterations, TS_NUM_BUCKETS, const_cast<char*>("iterations"), 1);
	  // debugPrintScreen(results.d_numLocks, TS_NUM_BUCKETS, const_cast<char*>("numLocks"), 1);
//		debugPrintScreen(results.d_maxMemUsage, TS_NUM_BUCKETS, "memory usage", 1);
//		debugPrintScreen(results.d_numIdle, TS_NUM_BUCKETS, "idle time", 1);
	  debugPrintScreen(results.d_errorMsg, TS_NUM_BUCKETS, const_cast<char*>("errors"), 1,0);

//		debugPrintScreen(d_debugData, 7*TS_NUM_BUCKETS, "debug Info", 7);
//		debugPrintScreen(d_locks, TS_NUM_BUCKETS, "lock Array", TS_NUM_BUCKETS);

	  debugPrintScreen(results.d_numDonations, TS_NUM_BUCKETS, const_cast<char*>("donations"), 1,0);
	  debugPrintScreen(results.d_numSteals, TS_NUM_BUCKETS, const_cast<char*>("steals"), 1,0);
	  int processed = debugPrintScreen(results.d_numPatchesProcessed, TS_NUM_BUCKETS, const_cast<char*>("patches"), 1,0);


	  int unprocessed = debugPrintScreen(d_numPatches, 1, const_cast<char*>("global numPatches"), 1,0);

	  float total = processed+unprocessed;
	  
//		debugPrintScreen(d_outputQueueLoc, 1, "outputQueueSize", 1);
	}
	else
	{
		// debugPrintScreen(results.d_numIterations, TS_NUM_BUCKETS, "iterations", 1);
		// debugPrintScreen(results.d_numLocks, TS_NUM_BUCKETS, "numLocks", 1);
//		debugPrintScreen(results.d_maxMemUsage, TS_NUM_BUCKETS, "memory usage", 1);
//		debugPrintScreen(results.d_numIdle, TS_NUM_BUCKETS, "idle time", 1);
//		debugPrintScreen(results.d_numSteals, TS_NUM_BUCKETS, "steals", 1);
//		debugPrintScreen(results.d_errorMsg, TS_NUM_BUCKETS, "errors", 1);

//		debugPrintScreen(d_debugData, 7*TS_NUM_BUCKETS, "debug Info", 7);
//		debugPrintScreen(d_locks, TS_NUM_BUCKETS, "lock Array", TS_NUM_BUCKETS);
		// debugPrintScreen(d_numPatches, 1, "global numPatches", 1);
//		debugPrintScreen(d_outputQueueLoc, 1, "outputQueueSize", 1);

	}
	cutilSafeCall(cudaFree(d_outputQueueLoc));
	cutilSafeCall(cudaFree(d_numPatches));

	return time;
//	printWorkQueue(results.d_workInQueue);
}//Launch_Synth_Scheduler

       
