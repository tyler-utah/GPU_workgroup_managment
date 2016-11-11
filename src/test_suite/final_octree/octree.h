DEFINE_int32(numParticles, 3000000, "number of particles to treat");
DEFINE_int32(maxChildren, 20, "maximum number of children");
DEFINE_int32(threads, 128, "number of threads");
DEFINE_int32(pool_size, 20000, "size of task pools");

// ---
DEFINE_string(restoration_ctx_path, "test_suite/final_octree/", "Path to restoration context");
DEFINE_string(merged_kernel_file, "test_suite/final_octree/device/merged.cl", "path to the merged mega kernel file");
DEFINE_string(persistent_kernel_file, "test_suite/final_octree/device/standalone.cl", "path to the standalone mega kernel file");

/*---------------------------------------------------------------------------*/

typedef struct {
  cl_float4 middle;
  cl_bool flip;
  cl_uint end;
  cl_uint beg;
  cl_uint treepos;
} Task;

/*---------------------------------------------------------------------------*/
// global vars

const unsigned int MAXTREESIZE = 50000000;
int num_workgroups = 0;

cl::Buffer particles;
cl::Buffer newParticles;
cl::Buffer tree;
cl::Buffer treeSize;
cl::Buffer particlesDone;
cl::Buffer pools;
cl::Buffer task_pool_lock;
cl::Buffer pool_head;
Task *h_pools = NULL;

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

/*---------------------------------------------------------------------------*/

const char* persistent_kernel_name() {
  return "octree_main";
}

/*---------------------------------------------------------------------------*/

void reset_persistent_task(CL_Execution *exec) {
  // re-write 0 to the CL buffers, etc
  int err = 0;

  err = exec->exec_queue.enqueueFillBuffer(tree, 0, 0, sizeof(cl_uint)*MAXTREESIZE);
  check_ocl(err);
  err = exec->exec_queue.enqueueWriteBuffer(pools, CL_TRUE, 0, num_workgroups * FLAGS_pool_size * sizeof(Task), h_pools);
  check_ocl(err);
  err = exec->exec_queue.enqueueFillBuffer(task_pool_lock, 0, false, num_workgroups * sizeof(cl_uint));
  check_ocl(err);
  err = exec->exec_queue.enqueueFillBuffer(pool_head, 0, 0, num_workgroups * sizeof(cl_uint));
  check_ocl(err);

  generate_particles(exec);
}

/*---------------------------------------------------------------------------*/

// Empty for Pannotia apps
void init_persistent_app_for_real(CL_Execution *exec, int occupancy) {
  num_workgroups = occupancy;

  particles = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_float4) * FLAGS_numParticles);
  newParticles = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_float4) * FLAGS_numParticles);
  tree = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint)*MAXTREESIZE);
  treeSize = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint));
  particlesDone = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint));

  pools = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_workgroups * FLAGS_pool_size * sizeof(Task));
  task_pool_lock = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_workgroups * sizeof(cl_uint));
  pool_head = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_workgroups * sizeof(cl_uint));

  h_pools = (Task *)calloc(num_workgroups * FLAGS_pool_size, sizeof(Task));
  if (h_pools == NULL) {
    cout << "calloc failed" << endl;
    exit(EXIT_FAILURE);
  }

  reset_persistent_task(exec);
}

/*---------------------------------------------------------------------------*/

void set_persistent_app_args_for_real(int arg_index, cl::Kernel k) {
  // Set args for persistent kernel

  check_ocl(k.setArg(arg_index++, particles));
  check_ocl(k.setArg(arg_index++, newParticles));
  check_ocl(k.setArg(arg_index++, tree));
  check_ocl(k.setArg(arg_index++, FLAGS_numParticles));
  check_ocl(k.setArg(arg_index++, treeSize));
  check_ocl(k.setArg(arg_index++, particlesDone));
  check_ocl(k.setArg(arg_index++, FLAGS_maxChildren));

  check_ocl(k.setArg(arg_index++, pools));
  check_ocl(k.setArg(arg_index++, task_pool_lock));
  check_ocl(k.setArg(arg_index++, pool_head));
  check_ocl(k.setArg(arg_index++, num_workgroups));
  check_ocl(k.setArg(arg_index++, FLAGS_pool_size));

  check_ocl(k.setArg(arg_index++, NULL));
  check_ocl(k.setArg(arg_index++, NULL));
}

/*---------------------------------------------------------------------------*/

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
  free(h_pools);
}

bool diff_solution_file_int(int * a, const char * solution_fname, int v) {
  // pannotia specific
  return 0;
}

bool check_persistent_task(CL_Execution *exec) {
  // check whether the output is correct, load cl buffer back in host
  // and check values
  cl_uint* htree;
  cl_uint htreeSize;
  cl_uint hparticlesDone;
  cl_int err = 0;

  err = exec->exec_queue.enqueueReadBuffer(particlesDone, CL_TRUE, 0, sizeof(cl_uint), &hparticlesDone);
  check_ocl(err);
  err = exec->exec_queue.enqueueReadBuffer(treeSize, CL_TRUE, 0, sizeof(cl_uint), &htreeSize);
  check_ocl(err);
  htree = new cl_uint[MAXTREESIZE];
  err = exec->exec_queue.enqueueReadBuffer(tree, CL_TRUE, 0, sizeof(cl_uint) * MAXTREESIZE, htree);
  check_ocl(err);

  unsigned int sum = 0;
  for(unsigned int i = 0; i < htreeSize; i++) {
    if (htree[i] & 0x80000000) {
      sum += htree[i] & 0x7fffffff;
    }
  }

  std::cout << "OCTREE: Particles in tree: " << sum << " (" << FLAGS_numParticles << ") [" << hparticlesDone << "]" << std::endl;
  delete(htree);

  // // For debug
  // cl_int *h_pool_head = (cl_int *)calloc(num_workgroups, sizeof(cl_int));
  // if (h_pool_head == NULL) {
  //   cout << "calloc failed" << endl;
  //   exit(EXIT_FAILURE);
  // }
  // err = exec->exec_queue.enqueueReadBuffer(pool_head, CL_TRUE, 0, num_workgroups * sizeof(cl_uint), h_pool_head);
  // check_ocl(err);
  // for (int i = 0; i < num_workgroups; i++) {
  //   printf("pool head %.3d: %u\n", i, h_pool_head[i]);
  // }
  // free(h_pool_head);

  return (sum == FLAGS_numParticles) && (hparticlesDone == FLAGS_numParticles);
}
