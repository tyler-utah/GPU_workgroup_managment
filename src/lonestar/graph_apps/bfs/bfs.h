#include "portable_endian.h"

// Could probably go into a pannotia header
DEFINE_string(graph_file, "", "Path to the graph_file");
DEFINE_string(graph_output, "", "Path to output the graph result");
DEFINE_string(graph_solution_file, "pannotia/solutions/sssp_usa.txt", "the path the mega kernel file");


// ---
DEFINE_string(restoration_ctx_path, "lonestar/graph_apps/bfs", "Path to restoration context");
DEFINE_string(merged_kernel_file, "lonestar/graph_apps/bfs/device/merged.cl", "the path the mega kernel file");
DEFINE_string(persistent_kernel_file, "lonestar/graph_apps/bfs/device/standalone.cl", "the path the mega kernel file");

#define MYINFINITY 1000000000

typedef struct {

	cl_uint nnodes, nedges;
	cl_uint *noutgoing, *psrc, *edgessrcdst;

} Graph;

cl::Buffer d_dist, d_in_wl, d_in_index, d_out_wl, d_out_index,  d_g_edgessrcdst, d_g_psrc;
cl_uint g_nnodes;
Graph hgraph;
cl_uint *zero_array;
cl_uint *read_array;
const float wl_mult = .25;
cl_int* h_wl;
cl_int h_index;

void * safe_malloc(size_t x) {
	void * tmp = malloc(x);
	if (!tmp) {
		check_ocl(-1);
	}
	return tmp;
}

unsigned allocOnHost(Graph * g) {
  g->edgessrcdst = (cl_uint *) safe_malloc((g->nedges+1) * sizeof(unsigned int)); // First entry acts as null.
  g->psrc = (cl_uint *) calloc(g->nnodes+1, sizeof(unsigned int));           // Init to null.
  g->psrc[g->nnodes] = g->nedges;                                            // Last entry points to end of edges, to avoid thread divergence in drelax.
  g->noutgoing = (cl_uint *) calloc(g->nnodes, sizeof(unsigned int));        // Init to 0.
  return 0;
}

unsigned readFromEdges(Graph * g, const char* file) {

  unsigned int prevnode = 0;
  unsigned int tempsrcnode;
  unsigned int ncurroutgoing = 0;

  std::ifstream cfile;
  cfile.open(file);

  std::string str;
  getline(cfile, str);
  sscanf(str.c_str(), "%d %d", &(g->nnodes), &(g->nedges));

  allocOnHost(g);
  for (unsigned ii = 0; ii < g->nnodes; ++ii) {
    //g->srcsrc[ii] = ii;
  }

  for (unsigned ii = 0; ii < g->nedges; ++ii) {
    getline(cfile, str);
    //sscanf(str.c_str(), "%d %d %d", &tempsrcnode, &(g->edgessrcdst[ii+1]), &(g->edgessrcwt[ii+1]));

    if (prevnode == tempsrcnode) {

      if (ii == 0) {
        g->psrc[tempsrcnode] = ii + 1;
      }
      ++ncurroutgoing;
    }
    else {
      g->psrc[tempsrcnode] = ii + 1;

      if (ncurroutgoing) {
        g->noutgoing[prevnode] = ncurroutgoing;
      }
      prevnode = tempsrcnode;
      ncurroutgoing = 1; // Not 0.
    }
    //g->nincoming[g->edgessrcdst[ii+1]]++;
    //progressPrint(g, g->nedges, ii);
  }
  g->noutgoing[prevnode] = ncurroutgoing; // Last entries.

  cfile.close();
  return 0;
}

unsigned readFromGR(Graph *g, const char *file) {

  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long masterLength = ftell(f);
  fseek(f, 0, SEEK_SET);

  void *m = (void *)malloc(masterLength + 1);
  int fread_ret = fread(m, masterLength, 1, f);

  if (fread_ret != 1) {
    printf("error in fread!!\n");
    abort();
  }
  fclose(f);

  double starttime, endtime;
  //starttime = rtclock();

  // Parse file
  uint64_t* fptr = (uint64_t*)m;
  uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  uint64_t numNodes = le64toh(*fptr++);
  uint64_t numEdges = le64toh(*fptr++);
  uint64_t *outIdx = fptr;
  fptr += numNodes;
  uint32_t *fptr32 = (uint32_t*)fptr;
  uint32_t *outs = fptr32;
  fptr32 += numEdges;
  if (numEdges % 2) fptr32 += 1;
  unsigned  *edgeData = (unsigned *)fptr32;

  g->nnodes = numNodes;
  g->nedges = numEdges;

  printf("nnodes=%d, nedges=%d.\n", g->nnodes, g->nedges);

  allocOnHost(g);

  // Get rid of srcsrc and replace with ii
  for (unsigned ii = 0; ii < g->nnodes; ++ii) {
    //g->srcsrc[ii] = ii;

	// Replace noutgoing[ii] with psrc[ii+1] - psrc[ii] - 1
    if (ii > 0) {
      g->psrc[ii] = le64toh(outIdx[ii - 1]) + 1;
      g->noutgoing[ii] = le64toh(outIdx[ii]) - le64toh(outIdx[ii - 1]);
    }
    else {
      g->psrc[0] = 1;
      g->noutgoing[0] = le64toh(outIdx[0]);
    }
    for (unsigned jj = 0; jj < g->noutgoing[ii]; ++jj) {
      unsigned edgeindex = g->psrc[ii] + jj;
      unsigned dst = le32toh(outs[edgeindex - 1]);

      if (dst >= g->nnodes)
        printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, edgeindex);

      g->edgessrcdst[edgeindex] = dst;
      //g->edgessrcwt[edgeindex] = edgeData[edgeindex - 1];

      //g->nincoming[dst]++;
    }
    //progressPrint(g, g->nnodes, ii);
  }

  //endtime = rtclock();

  //printf("read %ld bytes in %0.2f ms (%0.2f MB/s)\n", masterLength, 1000 * (endtime - starttime), (masterLength / 1048576) / (endtime - starttime));

  return 0;
}

// Reads a graph from a file and stores it
// in a host Graph structure
int read_graph(Graph* g, const char* file) {
  if (strstr(file, ".edges")) {
    return readFromEdges(g, file);
  } else if (strstr(file, ".gr")) {
    return readFromGR(g, file);
  }
  return 0;
}

// Free all the host memory for the graph
void free_host_graph(Graph *g) {
  free(g->edgessrcdst);
  //free(g->edgessrcwt);
  free(g->psrc);
  free(g->noutgoing);
  //free(g->nincoming);
  //free(g->srcsrc);
}

const char* persistent_app_name() {
  return "lonestar_bfs";
}

const char* persistent_kernel_name() {
  return "drelax2";
}

// Empty for Pannotia apps
void init_persistent_app_for_real(CL_Execution *exec, int occupancy) {
  return;
}

void set_persistent_app_args_for_real(int arg_index, cl::Kernel k) {
  return;
}

void init_persistent_app_for_occupancy(CL_Execution *exec) {

	cl_int err = 0;
  read_graph(&hgraph, FLAGS_graph_file.c_str());
  
  zero_array = (cl_uint *) malloc(sizeof(cl_uint) * hgraph.nnodes);
  if (!zero_array) {
	  check_ocl(-1);
  }
  read_array = (cl_uint *) malloc(sizeof(cl_uint) * hgraph.nnodes);
  if (!read_array) {
	  check_ocl(-1);
  }
  
  d_dist = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (hgraph.nnodes) * sizeof(cl_uint), nullptr, &err);
  check_ocl(err);
  d_in_index = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &err);
  check_ocl(err);
  d_out_index = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &err);
  check_ocl(err);
  
  // TYLER: Copied from OOPSLA code
  // Theoretically, the buffers should be size (hgraph.nedges * 2),
  // but some GPUs we tested do not have enough memory. Using less
  // memory works for all graphs and GPUs we've tested so far.  
  d_in_wl = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE,  ((hgraph.nedges)) * sizeof(cl_int), nullptr, &err);
  check_ocl(err);
  d_out_wl = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE,  ((hgraph.nedges)) * sizeof(cl_int), nullptr, &err);
  check_ocl(err);
  
  //d_g_noutgoing = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE,  sizeof(cl_uint), nullptr, &err);
  //check_ocl(err);
  d_g_edgessrcdst = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE,(hgraph.nedges + 1) * sizeof(cl_uint), nullptr, &err);
  check_ocl(err);
  //d_g_srcsrc = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
  //check_ocl(err);
  d_g_psrc = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE,(hgraph.nnodes + 1) * sizeof(cl_uint), nullptr, &err);
  check_ocl(err);
  
  g_nnodes = hgraph.nnodes;
  
  for (int i = 0; i < hgraph.nnodes; i++) {
	  zero_array[i] = MYINFINITY;
	  read_array[i] = MYINFINITY;
  }
  
  cl_int zero = 0;
  
  err = exec->exec_queue.enqueueWriteBuffer(d_dist, CL_TRUE, 0, (hgraph.nnodes) * sizeof(cl_int), zero_array);
  check_ocl(err);

  // First value of dist is initialised to 0
  err = exec->exec_queue.enqueueWriteBuffer(d_dist, CL_TRUE, 0, sizeof(cl_int), &zero);
  check_ocl(err);

  //err = exec->exec_queue.enqueueWriteBuffer(d_in_wl, CL_TRUE, 0, (hgraph.nnodes) * sizeof(cl_int), read_array);
  //check_ocl(err);
  //err = exec->exec_queue.enqueueWriteBuffer(d_out_wl, CL_TRUE, 0, (hgraph.nnodes) * sizeof(cl_int), read_array);
  //check_ocl(err);
  err = exec->exec_queue.enqueueWriteBuffer(d_in_index, CL_TRUE, 0, sizeof(cl_int), &zero);
  check_ocl(err);
  err = exec->exec_queue.enqueueWriteBuffer(d_out_index, CL_TRUE, 0, sizeof(cl_int), &zero);
  check_ocl(err);
  
  //err = exec->exec_queue.enqueueWriteBuffer(d_g_noutgoing, CL_TRUE, 0, (hgraph.nnodes) * sizeof(cl_uint), hgraph.noutgoing);
  check_ocl(err);
  err = exec->exec_queue.enqueueWriteBuffer(d_g_edgessrcdst, CL_TRUE, 0, (hgraph.nedges + 1) * sizeof(cl_uint), hgraph.edgessrcdst);
  check_ocl(err);
  //err = exec->exec_queue.enqueueWriteBuffer(d_g_srcsrc, CL_TRUE, 0, (hgraph.nnodes) * sizeof(cl_uint), hgraph.srcsrc);
  check_ocl(err);
  err = exec->exec_queue.enqueueWriteBuffer(d_g_psrc, CL_TRUE, 0, (hgraph.nnodes + 1) * sizeof(cl_uint), hgraph.psrc);
  check_ocl(err);
  
}

void reset_persistent_task(CL_Execution *exec) {
	
  for (int i = 0; i < hgraph.nnodes; i++) {
    zero_array[i] = MYINFINITY;
    read_array[i] = MYINFINITY;
  }
  
  cl_int zero = 0;
  int err = 0;
  
  err = exec->exec_queue.enqueueWriteBuffer(d_dist, CL_TRUE, 0, (hgraph.nnodes) * sizeof(cl_int), zero_array);
  check_ocl(err);

  // First value of dist is initialised to 0
  err = exec->exec_queue.enqueueWriteBuffer(d_dist, CL_TRUE, 0, sizeof(cl_int), &zero);
  check_ocl(err);
  err = exec->exec_queue.enqueueWriteBuffer(d_in_index, CL_TRUE, 0, sizeof(cl_int), &zero);
  check_ocl(err);
  err = exec->exec_queue.enqueueWriteBuffer(d_out_index, CL_TRUE, 0, sizeof(cl_int), &zero);
  check_ocl(err);
  
  //err = exec->exec_queue.enqueueWriteBuffer(d_g_noutgoing, CL_TRUE, 0, (hgraph.nnodes) * sizeof(cl_uint), hgraph.noutgoing);
  check_ocl(err);
  err = exec->exec_queue.enqueueWriteBuffer(d_g_edgessrcdst, CL_TRUE, 0, (hgraph.nedges + 1) * sizeof(cl_uint), hgraph.edgessrcdst);
  check_ocl(err);
  //err = exec->exec_queue.enqueueWriteBuffer(d_g_srcsrc, CL_TRUE, 0, (hgraph.nnodes) * sizeof(cl_uint), hgraph.srcsrc);
  check_ocl(err);
  err = exec->exec_queue.enqueueWriteBuffer(d_g_psrc, CL_TRUE, 0, (hgraph.nnodes + 1) * sizeof(cl_uint), hgraph.psrc);
  check_ocl(err);
}

int set_persistent_app_args_for_occupancy(int arg_index, cl::Kernel k) {
	cout << "arg index " << arg_index << endl;
  check_ocl(k.setArg(arg_index++, d_dist));
  check_ocl(k.setArg(arg_index++, d_in_index));
  check_ocl(k.setArg(arg_index++, d_out_index));
  check_ocl(k.setArg(arg_index++, d_in_wl));
  check_ocl(k.setArg(arg_index++, d_out_wl));
  check_ocl(k.setArg(arg_index++, g_nnodes));
  //check_ocl(k.setArg(arg_index++, d_g_noutgoing));
  check_ocl(k.setArg(arg_index++, d_g_edgessrcdst));
  //check_ocl(k.setArg(arg_index++, d_g_srcsrc));
  check_ocl(k.setArg(arg_index++, d_g_psrc));
  return arg_index;
}

void clean_persistent_task(CL_Execution *exec) {
  free(zero_array);
  free(read_array);
  free_host_graph(&hgraph);
}

bool diff_solution_file_int(cl_uint * a, const char * solution_fname, int v) {
	bool ret = true;
	FILE * fp = fopen(solution_fname, "r");

	// We start at 1 because 0 is a special case
	for (int i = 0; i < v; i++) {
		//printf("%d:%d\n", i, a[i]);
		if (feof(fp)) {
			printf("111\n");
			ret = false;
			break;
		}
		int compare;
		int trash;
		int found = fscanf(fp, "%d:%d\n", &trash, &compare);
		if (found != 2) {
			printf("222\n");
			ret = false;
			break;
		}
		if (compare != a[i]) {
			printf("%d found %d expected %d\n", i, compare, a[i]);
			ret = false;
			break;
		}
	}

	fclose(fp);
	return ret;
}

int check_persistent_task(CL_Execution *exec) {
	exec->exec_queue.enqueueReadBuffer(d_dist, CL_TRUE, 0, sizeof(cl_uint) * hgraph.nnodes, read_array);
	return diff_solution_file_int(read_array, FLAGS_graph_solution_file.c_str(), hgraph.nnodes);
}

