
#ifndef __SYNTHWORK_H
#define __SYNTHWORK_H

#include <stdio.h>
// defines
#include <cstdio>

#define NO_MORE_WORK 0

//structs to hold the results
struct SCHEDULE_RESULTS
{
	int * d_numIterations;	//number of iterations that each bin went through (size = TS_NUM_BUCKETS)
	int * d_numLocks;		//number of atmoic spin locks that were used (size = TS_NUM_BUCKETS)
	int * d_maxMemUsage;	//maximum amount of memory used (size = TS_NUM_BUCKETS)
	int * d_workInQueue;	//the amount of work in the queue after one iteration (size = TS_NUM_BUCKETS * TS_MAX_ITERATIONS)
	int * d_numSteals;		//the number of steal attempts made by a block
	int * d_numDonations;	//the number of donation attempts made by a block
	int * d_numIdle;		//the number of iterations a block was idle (could not grab work to do)
	int * d_errorMsg;		//error messages from the scheduler
	int * d_numPatchesTaken;//the number of patches taken from the dequeue 
	int * d_numPatchesProcessed;
};//end schedule results

// performs any initialization needed

void initSynthWork( float exit_probability, 
                    int nchildren, 
                    int worksize_when_producing, 
                    int worksize_when_consuming,
                    float *randbuffer,
                    int nrands
                    );

void deleteSynthWork( float exit_probability, 
                    int nchildren, 
                    int worksize_when_producing, 
                    int worksize_when_consuming,
                    float *randbuffer,
                    int nrands
                    );

float Launch_Synth_Scheduler(float * d_inputData, float * d_outputData, float * d_debugData,
							int * d_intDebugData, int * d_headQueuePtr, int * d_tailQueuePtr,
							int * d_locks, float * d_randData,int numElements,SCHEDULE_RESULTS results);

// this function occupies the GPU for 
// a while, then returns with a count of
// new work items needed.

__device__ inline int doSynthWork(float *dumbptr, float unifrand);

//prototypes for the cudaHelperFunc.cu go here
int debugPrintScreen(unsigned int * deviceData, unsigned int numElements, char * variableName, int stride);
int debugPrintScreen(int * deviceData, unsigned int numElements, char * variableName, int stride);
void debugPrintScreen(float * deviceData, unsigned int numElements, char * variableName, int stride);
void find_free_mem2();
void printWorkQueue(int * d_workQueue);

void writeOutLockUsage(int * d_locks, char * fileName, float p, float m);
void computeWorkImbalance(int * d_workQueue, FILE * outFile = stdout);
void writeOutIdle(int * d_idle, char * fileName, float p, float m);
void writeOutPatchesTaken(int * d_patchesTaken, char * fileName, float p, float m);
void writeOutPatchesProcessed(int * d_patchesProcessed, char * fileName, float p, float m);
void writeOutPerformance(int * d_patchesTaken, char * fileName, float p, float time);
void writeOutAvgLocksUsed(int * d_locks, char * fileName, float p, float m);
#endif
