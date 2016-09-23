/*
cudaHelperFunc.cu

all helper routines(memtests, debug prints, etc.) go in here
*/

#include <stdio.h>
#include <cuda_runtime_api.h>
//#include <cutil_inline.h>
//#include <cuda_gl_interop.h>

#include "globals.h"
#include "synthwork.h"

#define CUDA_SAFE_CALL checkCudaErrors

//-------------------------------------------------------------------------------------------------------
//some debugging functions to print stuff onto screen
int debugPrintScreen(unsigned int * deviceData, unsigned int numElements, char * variableName, int stride, int printall)
{
    printf("debugging to screen: %s\n\n", variableName);

    unsigned int * host_data = (unsigned int *)malloc(sizeof(unsigned int) * numElements);

    CUDA_SAFE_CALL(cudaMemcpy(host_data, deviceData,sizeof(unsigned int) * numElements, cudaMemcpyDeviceToHost));
    int total = 0;
    for(unsigned int i=0; i<numElements; i++)
    {
      if (printall) {
        if(i % stride == 0) printf("\n");
        printf("%u ", host_data[i]);
      }
      total += host_data[i];
    }
    printf("\n\n");
    printf("total: %d\n\n", total);
    free(host_data);
    return total;
}

int debugPrintScreen(int * deviceData, unsigned int numElements, char * variableName, int stride, int printall)
{
    printf("debugging to screen: %s\n\n", variableName);

    int * host_data = (int *)malloc(sizeof(int) * numElements);

    CUDA_SAFE_CALL(cudaMemcpy(host_data, deviceData,sizeof(int) * numElements, cudaMemcpyDeviceToHost));
    int total = 0;
    for(unsigned int i=0; i<numElements; i++)
    {
      if (printall) {
        if(i % stride == 0) printf("\n");
        printf("%d ", host_data[i]);
    }
	total += host_data[i];
    }
    printf("\n\n");
    printf("total: %d\n\n", total);
    free( host_data);
    return total;
}

void debugPrintScreen(float * deviceData, unsigned int numElements, char * variableName, int stride)
{
    printf("debugging to screen: %s\n\n", variableName);

    float * host_data = (float*) malloc(sizeof( float) * numElements);

    CUDA_SAFE_CALL(cudaMemcpy(host_data, deviceData,sizeof(float) * numElements, cudaMemcpyDeviceToHost));

    for(unsigned int i=0; i<numElements; i++)
    {
        if(i % stride == 0) printf("\n%d:", i/stride);
        printf("%3.2f ", host_data[i]);
    }
    printf("\n\n");

    free( host_data);
}

//-----------------------------------------------------------
//Functions to Query Memory
/*
void find_free_mem2()
{
    float* temp_mem[100]; int curid=0;
    unsigned int cursize=MAX_MEM, totalsize=0;


    cudaError_t err=cudaSuccess;

   while(cursize>(1024*1024) && curid<100) // Don't go lower than 1 MB
   {
        err = cudaMalloc((void**) (&temp_mem[curid]), cursize);
        if(!err)
        {


         //printf("%dM\n",cursize/(1024*1024));
         curid++;
         if(curid>=100) 
         {
            fprintf(stderr," -E- memory is too fragmented\n");
         }
            totalsize+=cursize;


        }
        else
        {
         cursize/=2;
        }
   }

   for(int i=0; i<curid; i++)
   {
      cudaFree(temp_mem[i]);
   }

    fprintf(stderr,"[%u MB free]\n",totalsize/(1024*1024));


}
*/
//prints out the work queue to screen:
void printWorkQueue(int * d_workQueue)
{
	int size = TS_NUM_BUCKETS*TS_MAX_ITERATIONS;
	int * h_workQueue;
	h_workQueue = new int[size];

	cutilSafeCall(cudaMemcpy(h_workQueue, d_workQueue, sizeof(int)*size, cudaMemcpyDeviceToHost));

	for(int i=0; i<TS_MAX_ITERATIONS; i++)
	{
		for(int j=0; j<TS_NUM_BUCKETS; j++)
		{
			printf("%d ", h_workQueue[j*TS_MAX_ITERATIONS + i]);
		}
		printf("\n");
	}
	delete [] h_workQueue;
}

//computes the work imbalance by max imbalance vs mean imbalance
void computeWorkImbalance(int * d_workQueue, FILE * outFile)
{
	int size = TS_NUM_BUCKETS*TS_MAX_ITERATIONS;
	int * h_workQueue;
	h_workQueue = new int[size];

	cutilSafeCall(cudaMemcpy(h_workQueue, d_workQueue, sizeof(int)*size, cudaMemcpyDeviceToHost));
	float meanLoad;
	int maxLoad;

	for(int i=0; i<TS_MAX_ITERATIONS; i++)
	{
		meanLoad = 0;
		maxLoad = -100;
		for(int j=0; j<TS_NUM_BUCKETS; j++)
		{
			meanLoad += h_workQueue[j*TS_MAX_ITERATIONS + i];
			maxLoad = (maxLoad < h_workQueue[j*TS_MAX_ITERATIONS + i]) ? h_workQueue[j*TS_MAX_ITERATIONS + i] : maxLoad;
		}

		meanLoad /= TS_NUM_BUCKETS;

		//write out this information
		fprintf(outFile, "%3.2f %d\n",meanLoad, maxLoad); 
	}
	delete [] h_workQueue;
}

void writeOutLockUsage(int * d_locks, char * fileName, float p, float m)
{
	int size = TS_NUM_BUCKETS;
	int * h_locks;
	h_locks = new int[size];

	cutilSafeCall(cudaMemcpy(h_locks, d_locks, TS_NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

	//appends to the current file
	FILE * f = fopen(fileName, "w");
	fprintf(f, "%3.3f %3.3f ", p,m);
	for(int i=0; i<TS_NUM_BUCKETS; i++)
		fprintf(f,"%d ", h_locks[i]);
	fprintf(f,"\n");
	fclose(f);

}

void writeOutIdle(int * d_idle, char * fileName, float p, float m)
{
	int size = TS_NUM_BUCKETS;
	int * h_idle;
	h_idle = new int[size];

	cutilSafeCall(cudaMemcpy(h_idle, d_idle, TS_NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

	//appends to the current file
	FILE * f = fopen(fileName, "w");
	fprintf(f, "%3.3f %3.3f ", p,m);
	for(int i=0; i<TS_NUM_BUCKETS; i++)
		fprintf(f,"%d ", h_idle[i]);
	fprintf(f,"\n");
	fclose(f);
}

void writeOutPatchesTaken(int * d_patchesTaken, char * fileName, float p, float m)
{
	int size = TS_NUM_BUCKETS;
	int * h_patchesTaken;
	h_patchesTaken = new int[size];

	cutilSafeCall(cudaMemcpy(h_patchesTaken, d_patchesTaken, TS_NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

	//appends to the current file
	FILE * f = fopen(fileName, "w");
	fprintf(f, "%3.3f %3.3f ", p,m);
	for(int i=0; i<TS_NUM_BUCKETS; i++)
		fprintf(f,"%d ", h_patchesTaken[i]);
	fprintf(f,"\n");
	fclose(f);
}

void writeOutPatchesProcessed(int * d_patchesProcessed, char * fileName, float p, float m)
{
	int size = TS_NUM_BUCKETS;
	int * h_patchesProcessed;
	h_patchesProcessed = new int[size];

	cutilSafeCall(cudaMemcpy(h_patchesProcessed, d_patchesProcessed, TS_NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

	//appends to the current file
	FILE * f = fopen(fileName, "w");
	fprintf(f, "%3.3f %3.3f ", p,m);
	for(int i=0; i<TS_NUM_BUCKETS; i++)
		fprintf(f,"%d ", h_patchesProcessed[i]);
	fprintf(f,"\n");
	fclose(f);
}

void writeOutPerformance(int * d_patchesTaken, char * fileName, float p, float time)
{
	int size = TS_NUM_BUCKETS;
	int * h_patchesTaken;
	h_patchesTaken = new int[size];

	cutilSafeCall(cudaMemcpy(h_patchesTaken, d_patchesTaken, TS_NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

	int totalPatches = 0;

	for(int i=0; i<TS_NUM_BUCKETS; i++)
		totalPatches+= h_patchesTaken[i];

	printf("time: %f totalPatches: %d\n", time, totalPatches);
	printf("writing out to: %s\n", fileName);
	double performance = (double)totalPatches/ (double)time ;
	//appends to the current file
	FILE * f = fopen(fileName, "a");
	fprintf(f, "%f,%lf %f %d\n", p,performance, time, totalPatches);	
	fclose(f);
}

void writeOutAvgLocksUsed(int * d_locks, char * fileName, float p, float m)
{
	int size = TS_NUM_BUCKETS;
	int * h_locks;
	h_locks = new int[size];

	cutilSafeCall(cudaMemcpy(h_locks, d_locks, TS_NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

	int locksum =0;

	for(int i=0; i<TS_NUM_BUCKETS; i++)
		locksum += h_locks[i];

	double avgLock = (double) locksum / double(TS_NUM_BUCKETS);
	//appends to the current file
	FILE * f = fopen(fileName, "a");
	fprintf(f, "%3.3f %3.3f ", p,m);
	for(int i=0; i<TS_NUM_BUCKETS; i++)
		fprintf(f,"%3.2lf ", avgLock);
	fprintf(f,"\n");
	fclose(f);

	delete [] h_locks;
}
