/*
globals.h

add globals for both controlling CUDA and cpp compilation here
*/

#ifndef GLOBALS_H_

#define MAX_ARR (4000000)
#define MAX_MEM (2*1024*1024*1024) // Lets start with 2GB
#define SOME_PRIME (14331)

#define NUM_ELEMENTS 32	//keep this to a multiple of the bucket size
#define START_DATA_SIZE 512//	64 floats each
//use these defines to control which scheduler to run
//defaults to block scheduler
#define RUN_BLOCK 0
#define RUN_TASKSTEAL 1
#define RUN_TASKDONATE 2

//two defines for the locks in the case of block queuing
#define HEAD_LOCK 0
#define TAIL_LOCK 1

//defines for the size of buckets and the lot
#define TS_BUCKET_SIZE 6400
#define TS_NUM_BUCKETS 70
#define TS_MAX_ITERATIONS 200

//defines for the size of the block queue
#define BLOCKQUEUE_SIZE MAX_ARR

//settings for the synth work generator goes here
#define NUM_RANDS 2000	//anjul change this value if you think it is not enough / too much
#define SYNTH_EXIT_PROB 0.24
#define SYNTH_NUM_CHILDREN 4
#define SYNTH_PROD_WORK 0
#define SYNTH_CONS_WORK 0


//defines for some error messages
#define SCHED_ERROR_ALL_FULL 1

//externs
extern int scheduler;
#endif
