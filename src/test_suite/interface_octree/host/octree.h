#include "octree_types.h"

/*---------------------------------------------------------------------------*/

DEFINE_int32(numParticles, 10000, "number of particles to treat");
DEFINE_int32(maxChildren, 20, "maximum number of children");
DEFINE_int32(threads, 64, "number of threads");
//DEFINE_int32(num_iterations, 300, "number of iterations");

// see some octree types definitions in common/octree.h

// ---
DEFINE_string(restoration_ctx_path, "test_suite/interface_octree/common", "Path to restoration context");
DEFINE_string(merged_kernel_file, "test_suite/interface_octree/device/merged.cl", "path to the merged mega kernel file");
DEFINE_string(persistent_kernel_file, "test_suite/interface_octree/device/standalone.cl", "the path the mega kernel file");

/*---------------------------------------------------------------------------*/
// global vars

const unsigned int MAXTREESIZE = 11000000;
const unsigned int maxlength = 256;
int max_workgroups = 0;

//cl::Buffer d_num_iterations;
cl::Buffer randdata;
cl::Buffer particles;
cl::Buffer newParticles;
cl::Buffer tree;
cl::Buffer treeSize;
cl::Buffer particlesDone;
cl::Buffer stealAttempts;
cl::Buffer deq;
cl::Buffer dh;
IW_barrier octree_h_bar;
cl::Buffer octree_d_bar;

/*---------------------------------------------------------------------------*/
// Specific to octree

// genrand_real1() is defined in rand.cc
double genrand_real1(void);

/*---------------------------------------------------------------------------*/

void generate_particles(CL_Execution *exec)
{
  cl_float4* lparticles = new cl_float4[FLAGS_numParticles];
  char fname[256];
  snprintf(fname, 256, "octreecacheddata-%dparticles.dat", FLAGS_numParticles);
  FILE* f = fopen(fname, "rb");
  if (!f) {
    cout << "Generating and caching data" << endl;

    int clustersize = 100;
    for (int i = 0; i < (FLAGS_numParticles / clustersize); i++) {
      float x = ((float)genrand_real1()*800.0f-400.0f);
      float y = ((float)genrand_real1()*800.0f-400.0f);
      float z = ((float)genrand_real1()*800.0f-400.0f);

      for (int x = 0; x < clustersize; x++) {
        lparticles[i*clustersize+x].s[0] = x + ((float)genrand_real1()*100.0f-50.0f);
        lparticles[i*clustersize+x].s[1] = y + ((float)genrand_real1()*100.0f-50.0f);
        lparticles[i*clustersize+x].s[2] = z + ((float)genrand_real1()*100.0f-50.0f);
      }
    }

    FILE* f = fopen(fname,"wb");
    fwrite(lparticles,sizeof(cl_float4), FLAGS_numParticles,f);
    fclose(f);
  } else {
    cout << "Read particle data from a file" << endl;
    fread(lparticles,sizeof(cl_float4), FLAGS_numParticles,f);
    fclose(f);
  }

  exec->exec_queue.enqueueWriteBuffer(particles, CL_TRUE, 0, sizeof(cl_float4) * FLAGS_numParticles, lparticles);
  delete lparticles;
}

/*---------------------------------------------------------------------------*/

const char* persistent_app_name() {
  return "octree";
}

const char* persistent_kernel_name() {
  return "octree_main";
}
/*---------------------------------------------------------------------------*/

void reset_persistent_task(CL_Execution *exec) {
  // re-write 0 to the CL buffers, etc
  int err = 0;

  for (int i = 0; i < MAX_P_GROUPS; i++) {
    octree_h_bar.barrier_flags[i] = 0;
  }
  octree_h_bar.phase = 0;
  // for sense reversal barrier
  octree_h_bar.counter = 0;
  octree_h_bar.sense = 0;

  err = exec->exec_queue.enqueueWriteBuffer(octree_d_bar, CL_TRUE, 0, sizeof(IW_barrier), &octree_h_bar);
  check_ocl(err);

  // err = exec->exec_queue.enqueueWriteBuffer(d_num_iterations, CL_TRUE, 0, sizeof(cl_int), &(num_iterations));
  // check_ocl(err);
  err = exec->exec_queue.enqueueFillBuffer(tree, 0, 0, sizeof(cl_uint)*MAXTREESIZE);
  check_ocl(err);
  err = exec->exec_queue.enqueueFillBuffer(deq, 0, 0, sizeof(Task) * maxlength * max_workgroups);
  check_ocl(err);
  err = exec->exec_queue.enqueueFillBuffer(dh, 0, 0, sizeof(DequeHeader) * max_workgroups);
  check_ocl(err);

  generate_particles(exec);
}

/*---------------------------------------------------------------------------*/

// Empty for Pannotia apps
void init_persistent_app_for_real(CL_Execution *exec, int occupancy) {
  int err = 0;
  max_workgroups = occupancy - 1;

  //d_num_iterations = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
  randdata = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int) * 128);
  particles = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_float4) * FLAGS_numParticles);
  newParticles = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_float4) * FLAGS_numParticles);
  tree = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint)*MAXTREESIZE);
  treeSize = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint));
  particlesDone = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint));
  stealAttempts = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint));
  deq = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(Task) * maxlength * max_workgroups);
  dh = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(DequeHeader) * max_workgroups);

  octree_d_bar = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(IW_barrier));

  reset_persistent_task(exec);

}

void set_persistent_app_args_for_real(int arg_index, cl::Kernel k) {
  // Set args for persistent kernel
  int err = 0;

  err = k.setArg(arg_index, randdata);
  arg_index++;
  check_ocl(err);

  err = k.setArg(arg_index, particles);
  arg_index++;
  check_ocl(err);

  err = k.setArg(arg_index, newParticles);
  arg_index++;
  check_ocl(err);
  err = k.setArg(arg_index, tree);
  arg_index++;
  check_ocl(err);
  err = k.setArg(arg_index, FLAGS_numParticles);
  arg_index++;
  check_ocl(err);
  err = k.setArg(arg_index, treeSize);
  arg_index++;
  check_ocl(err);
  err = k.setArg(arg_index, particlesDone);
  arg_index++;
  check_ocl(err);
  err = k.setArg(arg_index, FLAGS_maxChildren);
  arg_index++;
  check_ocl(err);

  err = k.setArg(arg_index, max_workgroups);
  arg_index++;
  check_ocl(err);
  err = k.setArg(arg_index, deq);
  arg_index++;
  check_ocl(err);
  err = k.setArg(arg_index, dh);
  arg_index++;
  check_ocl(err);
  err = k.setArg(arg_index, maxlength);
  arg_index++;
  check_ocl(err);

  /* frompart */
  err = k.setArg(arg_index, NULL);
  arg_index++;
  check_ocl(err);

  /* topart */
  err = k.setArg(arg_index, NULL);
  arg_index++;
  check_ocl(err);

}


void init_persistent_app_for_occupancy(CL_Execution *exec)
{
  // nothing to do for octree
}

int set_persistent_app_args_for_occupancy(int arg_index, cl::Kernel k) {
  // Set dummy args for persistent kernel
  int err = 0;
  int num_args_octree = 14;
  for (int i = 0; i < num_args_octree; i++) {
    err = k.setArg(arg_index, NULL);
    check_ocl(err);
    arg_index++;
  }
  return arg_index;
}

void output_persistent_solution(const char *fname, CL_Execution *exec) {
  // write to a file, nothing to do for octree
  return;
}

void clean_persistent_task(CL_Execution *exec) {
  // free malloc'ed values
}

bool diff_solution_file_int(int * a, const char * solution_fname, int v) {
  // pannotia specific
  return 0;
}

bool check_persistent_task(CL_Execution *exec) {
  // check whether the output is correct, load cl buffer back in host
  // and check values
  unsigned int* htree;
  unsigned int htreeSize;
  unsigned int hparticlesDone;
  int err = 0;

  err = exec->exec_queue.enqueueReadBuffer(particlesDone, CL_TRUE, 0, sizeof(cl_uint), &hparticlesDone);
  check_ocl(err);
  err = exec->exec_queue.enqueueReadBuffer(treeSize, CL_TRUE, 0, sizeof(cl_uint), &htreeSize);
  check_ocl(err);
  htree = new unsigned int[MAXTREESIZE];
  err = exec->exec_queue.enqueueReadBuffer(tree, CL_TRUE, 0, sizeof(cl_uint) * MAXTREESIZE, htree);
  check_ocl(err);

  unsigned int sum = 0;
  for(unsigned int i = 0; i < htreeSize; i++) {
    if (htree[i] & 0x80000000) {
      sum += htree[i] & 0x7fffffff;
    }
  }

  std::cout << "OCTREE: Particles in tree: " << sum << " (" << FLAGS_numParticles << ") [" << hparticlesDone << "]" << std::endl;

  if (sum == FLAGS_numParticles && hparticlesDone == FLAGS_numParticles) {
    return true;
  } else {
    return false;
  }
}
