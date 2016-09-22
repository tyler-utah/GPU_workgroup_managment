#include <CL/cl.h>
#include "stdio.h"
#include <stdlib.h>
#include "string.h"

#define STRINGIFY_INNER(X) #X
#define STRINGIFY(X) STRINGIFY_INNER(X)

void check_ocl_error(const int e, const char *file, const int line) {

  if (e < 0) {
    fprintf(stderr, "%s:%d: error (%d)\n", file, line, e);
    exit(1);
  }

}

#define check_ocl(err) check_ocl_error(err, __FILE__, __LINE__)

//Define these in command line if you'd like different
#ifndef MAX_DEVICES
#define MAX_DEVICES 10
#endif

#ifndef MAX_PLATFORMS
#define MAX_PLATFORMS 10
#endif

#define EXIT_FAILURE 1
#define SAFE_CALL(call) do {                     \
int SAFE_CALL_ERR = call;                        \
if(SAFE_CALL_ERR < 0) {                          \
  printf("error in file '%s' in line %i\n",      \
          __FILE__, __LINE__);                   \
  exit(EXIT_FAILURE);                            \
 } } while (0)


cl_device_id create_device(unsigned int platform_id, unsigned int device_id) {

   cl_platform_id platforms[MAX_PLATFORMS];
   cl_device_id dev[MAX_DEVICES];
   cl_uint num_devices;
   cl_uint num_platforms;


   //Should likely do some checks here to make sure platform_id is
   //less than 10 and less than the number of platforms returned

   /* Identify a platform */
   SAFE_CALL(clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms));

   if(platform_id >= num_platforms) {
     printf("requested platform id not available\n");
     exit(1);
   }

   /* Access a device */
   SAFE_CALL(clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, MAX_DEVICES, dev, &num_devices));

   if(device_id >= num_devices) {
     printf("requested device id not available\n");
     exit(1);
   }

   return dev[device_id];
}


cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename, const char* options) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err = 0;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1,
      (const char** )&program_buffer, &program_size, &err);
   check_ocl(err);   

   /* Build program */
   err = clBuildProgram(program, 1, &dev, options, NULL, NULL);
   if(err < 0) {

      // Find size of log and print to std output
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char* ) malloc(log_size + 1);

      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   free(program_buffer);
   return program;
}

void dump_binary(cl_program * program, const char * filename) {
   FILE *program_handle;
  program_handle = fopen(filename, "w");
   if(program_handle == NULL) {
      perror("Couldn't open binary file");
      exit(1);
   }
   size_t binary_sizes, num_devices;
   char ** output;
   int err;
   err = clGetProgramInfo(*program, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &num_devices, NULL);
   check_ocl(err);

   err = clGetProgramInfo(*program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_sizes, NULL);
   check_ocl(err);

   output = (char**)malloc(sizeof(char* )*1);
   output[0] = (char*)malloc(binary_sizes+1);

   err = clGetProgramInfo( *program, CL_PROGRAM_BINARIES, binary_sizes+1, output, NULL);
   check_ocl(err);
   output[0][binary_sizes] = '\0';
   fprintf(program_handle, "%s\n", output[0]);
   fclose(program_handle);
   free(output[0]);
   free(output);
   return;
}

void print_device_info(cl_device_id *device) {
  char buffer[512];
  cl_uint buf_uint;
  cl_ulong buf_ulong;
  cl_bool unified;
  printf("\n  -- device info --\n");
  clGetDeviceInfo(*device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
  printf("DEVICE_NAME:                   %s\n", buffer);
  clGetDeviceInfo(*device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
  printf("DEVICE_VENDOR:                 %s\n", buffer);
  clGetDeviceInfo(*device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
  printf("DEVICE_VERSION:                %s\n", buffer);
  clGetDeviceInfo(*device, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
  printf("DRIVER_VERSION:                %s\n", buffer);
  clGetDeviceInfo(*device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
  printf("DEVICE_MAX_COMPUTE_UNITS:      %u\n", (unsigned int)buf_uint);
  clGetDeviceInfo(*device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
  printf("DEVICE_MAX_CLOCK_FREQUENCY:    %u\n", (unsigned int)buf_uint);
  clGetDeviceInfo(*device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
  printf("DEVICE_GLOBAL_MEM_SIZE:        %llu\n", (unsigned long long)buf_ulong);
  clGetDeviceInfo(*device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
  printf("DEVICE_LOCAL_MEM_SIZE:         %llu\n", (unsigned long long)buf_ulong);
  clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
  printf("DEVICE_MAX_WORK_GROUP_SIZE:    %llu\n", (unsigned long long)buf_ulong);
  clGetDeviceInfo(*device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified, NULL);
  printf("CL_DEVICE_HOST_UNIFIED_MEMORY: %s\n", (unified)? "True" : "False");


}
