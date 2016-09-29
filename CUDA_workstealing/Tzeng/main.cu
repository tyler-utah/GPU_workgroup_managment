/*
this is the synthetic work launcher
Stan
*/

//INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
//#include <GL/glew.h>
//#include <GL/glut.h>
#include <cmath>

// CUDA headers
//#include <cuda_runtime_api.h>
//#include <cuda_gl_interop.h>
//#include <cutil.h>
//#include <cutil_inline.h>
//#include <cutil_gl_error.h> 

#include "helper_cuda.h"
#include "helper_string.h"

#include "globals.h"
#include "synthwork.h"

#define cutilSafeCall checkCudaErrors

#include "synthwork.cu"

using namespace std;

//FUNCTION PROTOTYPES
void initCUDA();
void runScheduler();
void freeCUDA();


void generateTaskStealData();
void generateBlockQueueData();
//GLOBALS

//used for CUDA work
float * d_inputData;
float * d_outputData;
float * d_debugData;
int   * d_intDebugData;
float * d_randData;

//for block queue, this will be sizeof(float)
//for task steal / donate this will be sizeof(float) * TS_NUM_BUCKETS
int * d_headQueuePtr;
int * d_tailQueuePtr;
int * d_locks;

//Tyler: I'm interested in donate :)
int scheduler = RUN_TASKDONATE;

SCHEDULE_RESULTS scheduler_results;

//command line arguments
float p;		//prob of a split
float m;		//number of children
float w_prod;	//work when producing
float w_cons;	//work when consuming

char * lockFileName = NULL;
char * iterFileName = NULL;	//for the output file
char * idleFileName = NULL;
char * patchFileName = NULL;
char * perfFileName = NULL;
char * processFileName = NULL;


//FUNCTIONS
int main(int argc,char ** argv) {
  //set up default commands
  p = SYNTH_EXIT_PROB;
  m = SYNTH_NUM_CHILDREN;
  w_prod = SYNTH_PROD_WORK;
  w_cons = SYNTH_CONS_WORK;
  //parse command line arguments here
  //  p = getCmdLineArgumentFloat(argc, (const char **)argv, "p");
  //m = getCmdLineArgumentFloat(argc, (const char **)argv, "m");
  //w_prod = getCmdLineArgumentFloat(argc, (const char **)argv, "w_prod");
  //w_cons = getCmdLineArgumentFloat(argc, (const char **)argv, "w_cons");
  
  
  //scheduler = getCmdLineArgumentInt(argc, (const char **) argv, "run");
  
  getCmdLineArgumentString(argc, (const char **) argv, "lockFileName", (char**) &lockFileName);
  getCmdLineArgumentString(argc, (const char **) argv, "iterFileName", (char**) &iterFileName);
  getCmdLineArgumentString(argc, (const char **) argv, "idleFileName", (char**) &idleFileName);
  getCmdLineArgumentString(argc, (const char **) argv, "patchFileName", (char**) &patchFileName);
  getCmdLineArgumentString(argc, (const char **) argv, "perfFileName", (char**) &perfFileName);
  getCmdLineArgumentString(argc, (const char **) argv, "processFileName", (char**) &processFileName);

  //init here
  initCUDA();
  
  //launch scheduler
  float time = Launch_Synth_Scheduler(d_inputData, d_outputData, d_debugData, d_intDebugData,
				      d_headQueuePtr, d_tailQueuePtr, d_locks, d_randData, START_DATA_SIZE*64,scheduler_results);
  
  printf("writing out data to files...\n");
  
  
  if(perfFileName != NULL) {
    printf("writing out performance data\n");
    writeOutPerformance(scheduler_results.d_numPatchesTaken,perfFileName,p,time);
    freeCUDA();
    return 0;
  }
  //debug usage only
  //	strcpy(lockFileName, "block_lock_results.txt");
  //	strcpy(idleFileName, "block_idle_stats.txt");
  //	writeOutLocks = 1;
  //	printf("iterFile: %s\n", iterFileName);
  printf("idleFile: %s\n", idleFileName);
  printf("lockFile: %s\n", lockFileName);
  
  //check to see what we write out
  if(iterFileName != NULL)
    {
      FILE * f = fopen(iterFileName, "w");
      
      fclose(f);
    }
  
  if(idleFileName != NULL)
    writeOutIdle(scheduler_results.d_numIdle,idleFileName,p,m); 
  
  if(lockFileName != NULL)
    writeOutLockUsage(scheduler_results.d_numLocks, lockFileName, p,m);
  
  if(patchFileName != NULL)
    writeOutPatchesTaken(scheduler_results.d_numPatchesTaken, patchFileName, p,m);
  
  //	if(processFileName != NULL)
  //		writeOutPatchesProcessed(scheduler_results.d_numPatchesProcessed, patchFileName, p,m);
  //free data
  freeCUDA();
  
  return 0;
}

void initCUDA() {
  //basic cuda inits
  if(scheduler == RUN_BLOCK) {

    cutilSafeCall(cudaMalloc((void**)&d_outputData, sizeof(float) * MAX_ARR));
    cutilSafeCall(cudaMalloc((void**)&d_debugData, sizeof(float) * MAX_ARR));
    cutilSafeCall(cudaMalloc((void**)&d_intDebugData, sizeof(int) * MAX_ARR));
    
    cutilSafeCall(cudaMalloc((void**)&d_headQueuePtr, sizeof(int) * 1));
    cutilSafeCall(cudaMalloc((void**)&d_tailQueuePtr, sizeof(int) * 1));
    cutilSafeCall(cudaMalloc((void**)&d_locks, sizeof(int) * 2));	//2 locks for head and tail
    
    //initialize the data inside the arrays here
    cutilSafeCall(cudaMemset(d_locks, 0, sizeof(int) * 2));
    cutilSafeCall(cudaMemset(d_headQueuePtr, 0, sizeof(int)));
    cutilSafeCall(cudaMemset(d_tailQueuePtr, 0, sizeof(int)));
    
    generateBlockQueueData();
  }
  else {
    //task steal / donate variation here
    cutilSafeCall(cudaMalloc((void**)&d_inputData, sizeof(float) * TS_NUM_BUCKETS * TS_BUCKET_SIZE));
    cutilSafeCall(cudaMalloc((void**)&d_outputData, sizeof(float) * TS_NUM_BUCKETS * TS_BUCKET_SIZE));
    cutilSafeCall(cudaMalloc((void**)&d_debugData, sizeof(float) * MAX_ARR));
    cutilSafeCall(cudaMalloc((void**)&d_intDebugData, sizeof(int) * TS_NUM_BUCKETS * TS_BUCKET_SIZE));
    
    cutilSafeCall(cudaMalloc((void**)&d_headQueuePtr, sizeof(int) * TS_NUM_BUCKETS));
    cutilSafeCall(cudaMalloc((void**)&d_tailQueuePtr, sizeof(int) * TS_NUM_BUCKETS));
    cutilSafeCall(cudaMalloc((void**)&d_locks, sizeof(int) * TS_NUM_BUCKETS));
    
    //initialize the data inside the arrays here
    cutilSafeCall(cudaMemset(d_locks, 0, sizeof(int) * TS_NUM_BUCKETS));
    cutilSafeCall(cudaMemset(d_headQueuePtr, 0, sizeof(int)*TS_NUM_BUCKETS));
    cutilSafeCall(cudaMemset(d_tailQueuePtr, 0, sizeof(int)*TS_NUM_BUCKETS));
    
    generateTaskStealData();
  }
  
  
  cutilSafeCall(cudaMalloc((void**)&d_randData, sizeof(float) * NUM_RANDS));
  
  //now set up the synth work 

  initSynthWork(p,m,w_prod,w_cons, d_randData,NUM_RANDS);
  
  //now set up the arrays to hold the resulting data
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_numIterations, sizeof(int) * TS_NUM_BUCKETS));
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_numLocks, sizeof(int) * TS_NUM_BUCKETS));
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_maxMemUsage, sizeof(int) * TS_NUM_BUCKETS));
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_workInQueue, sizeof(int) * TS_NUM_BUCKETS*TS_MAX_ITERATIONS));
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_numSteals, sizeof(int) * TS_NUM_BUCKETS));
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_numDonations, sizeof(int) * TS_NUM_BUCKETS));
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_numIdle, sizeof(int) * TS_NUM_BUCKETS));
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_errorMsg, sizeof(int) * TS_NUM_BUCKETS));
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_numPatchesTaken,sizeof(int) * TS_NUM_BUCKETS));
  cutilSafeCall(cudaMalloc((void**)&scheduler_results.d_numPatchesProcessed, sizeof(int) * TS_NUM_BUCKETS));
  
  
  cutilSafeCall(cudaMemset(scheduler_results.d_numSteals, 0,sizeof(int)*TS_NUM_BUCKETS));
  cutilSafeCall(cudaMemset(scheduler_results.d_numDonations, 0,sizeof(int)*TS_NUM_BUCKETS));
  cutilSafeCall(cudaMemset(scheduler_results.d_numIdle, 0,sizeof(int)*TS_NUM_BUCKETS));
  cutilSafeCall(cudaMemset(scheduler_results.d_errorMsg, 0,sizeof(int)*TS_NUM_BUCKETS));
  cutilSafeCall(cudaMemset(scheduler_results.d_numPatchesProcessed, 0,sizeof(int)*TS_NUM_BUCKETS));
}


void freeCUDA()
{
  cutilSafeCall(cudaFree(d_inputData));
  cutilSafeCall(cudaFree(d_outputData));
  cutilSafeCall(cudaFree(d_debugData));
  cutilSafeCall(cudaFree(d_intDebugData));
  cutilSafeCall(cudaFree(d_headQueuePtr));
  cutilSafeCall(cudaFree(d_tailQueuePtr));
  cutilSafeCall(cudaFree(d_locks));
  cutilSafeCall(cudaFree(d_randData));
  
  cutilSafeCall(cudaFree(scheduler_results.d_numIterations));
  cutilSafeCall(cudaFree(scheduler_results.d_numLocks));
  cutilSafeCall(cudaFree(scheduler_results.d_maxMemUsage));
  cutilSafeCall(cudaFree(scheduler_results.d_workInQueue));
  cutilSafeCall(cudaFree(scheduler_results.d_numSteals));
  cutilSafeCall(cudaFree(scheduler_results.d_numDonations));
  cutilSafeCall(cudaFree(scheduler_results.d_numIdle));
  cutilSafeCall(cudaFree(scheduler_results.d_errorMsg));
  cutilSafeCall(cudaFree(scheduler_results.d_numPatchesTaken));
  cutilSafeCall(cudaFree(scheduler_results.d_numPatchesProcessed));
}

void generateBlockQueueData()
{
  int totalData = START_DATA_SIZE * 64;
  
  float * h_inputData = new float[totalData];
  
  //load it with some random data
  for(int i=0; i<totalData; i++) {
    h_inputData[i] = 1.0;
  }
  
  cutilSafeCall(cudaMalloc((void**)&d_inputData, sizeof(float) * MAX_ARR));  
  //copy the data over
  cutilSafeCall(cudaMemset(d_inputData,0,sizeof(float)*MAX_ARR));
  cutilSafeCall(cudaMemcpy(d_inputData,h_inputData, sizeof(float) * totalData, cudaMemcpyHostToDevice));
  
  cutilSafeCall(cudaMemset(d_tailQueuePtr, 0, sizeof(int)));
  cutilSafeCall(cudaMemcpy(d_headQueuePtr, &totalData, sizeof(int), cudaMemcpyHostToDevice));
  delete [] h_inputData;
}//end generateBlockQueueData

void generateTaskStealData()
{
  float * h_tempBucket;
  h_tempBucket = new float [TS_BUCKET_SIZE * TS_NUM_BUCKETS];
  
  memset(h_tempBucket,0,sizeof(float) * TS_BUCKET_SIZE * TS_NUM_BUCKETS);
  
  int * h_dequeHeadPtr, * h_dequeTailPtr;
  h_dequeHeadPtr = new int[TS_NUM_BUCKETS];
  h_dequeTailPtr = new int[TS_NUM_BUCKETS];
  
  //load everything into CPU buckets first
  //first calculate the size of each individual bucket

  int numPatches = START_DATA_SIZE;
  
  //first, set the head starts, before any element is added, head = tail
  for(int i=0; i<TS_NUM_BUCKETS; i++)
    {
      //h_dequeHeadPtr[i] = h_dequeTailPtr[i] = counter;
      h_dequeHeadPtr[i] = h_dequeTailPtr[i] = 0;
      //		counter += bucketSize;
    }

  
  //now load the data into buckets, recording the end location along the way
  int bucketStartIndex=0;
  int currBucket=0;
  for(int i=0; i<numPatches; i++)
    {
      bucketStartIndex = h_dequeHeadPtr[currBucket];
      for(int j=0; j<64; j++)
	h_tempBucket[bucketStartIndex+j] = 1.0;
      h_dequeHeadPtr[currBucket] +=64;
      currBucket ++;
      currBucket %= TS_NUM_BUCKETS;
    }
  
  // for(int i=0; i<TS_NUM_BUCKETS;i++)
  //   printf("tail: %d  head: %d\n", h_dequeTailPtr[i], h_dequeHeadPtr[i]);
  
  
  //now copy the data onto GPU memory
  cutilSafeCall(cudaMemcpy(d_inputData, h_tempBucket, sizeof(float)*TS_BUCKET_SIZE * TS_NUM_BUCKETS, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(d_tailQueuePtr, h_dequeTailPtr, sizeof(int)*TS_NUM_BUCKETS, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(d_headQueuePtr, h_dequeHeadPtr, sizeof(int)*TS_NUM_BUCKETS , cudaMemcpyHostToDevice));
  
  
  delete [] h_tempBucket;
  delete [] h_dequeHeadPtr;
  delete [] h_dequeTailPtr;
}
