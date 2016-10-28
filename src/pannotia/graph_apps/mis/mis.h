#include "parse.h"

// Could probably go into a pannotia header
DEFINE_string(graph_file, "", "Path to the graph_file");
DEFINE_string(graph_output, "", "Path to output the graph result");
DEFINE_string(graph_solution_file, "pannotia/solutions/mis_ecology.txt", "to the solution graph file");


// ---
DEFINE_string(restoration_ctx_path, "pannotia/graph_apps/mis", "Path to restoration context");
DEFINE_string(merged_kernel_file, "pannotia/graph_apps/mis/device/merged.cl", "the path the mega kernel file");
DEFINE_string(persistent_kernel_file, "pannotia/graph_apps/mis/device/standalone.cl", "the path the mega kernel file");

int num_nodes = 0, num_edges = 0;
csr_array *csr;
cl_int *s_array_input;
cl_int *s_array_output;
cl_int *zeros, *n_ones;
cl_float *node_value;
cl::Buffer row_d, col_d, stop_d, s_array_d, node_value_d, min_array_d, c_array_d, c_u_array_d;

const char* persistent_app_name() {
	return "mis";
}

const char* persistent_kernel_name() {
	return "mis_combined";
}

// Empty for Pannotia apps
void init_persistent_app_for_real(CL_Execution *exec, int occupancy) {
	return;
}

void set_persistent_app_args_for_real(int arg_index, cl::Kernel k) {
	return;
}


void init_persistent_app_for_occupancy(CL_Execution *exec) {

	csr = parseMetis(FLAGS_graph_file.c_str(), &num_nodes, &num_edges, 0);
	
	s_array_output = (cl_int *) malloc(num_nodes * sizeof(cl_int));
	zeros = (cl_int *)malloc(num_nodes * sizeof(cl_int));
	n_ones = (cl_int *)malloc(num_nodes * sizeof(cl_int));
	node_value = (cl_float *)malloc(num_nodes * sizeof(cl_float));	

	
	srand(6);
	for (int i = 0; i < num_nodes; i++) {
		zeros[i] = 0;
		n_ones[i] = -1;
		// Original application: Node_value[i] =  rand()/(float)RAND_MAX;
		node_value[i] = i / (float)(num_nodes + 1);
		//node_value[i] = rand() / (float)RAND_MAX;
	}

	//cl_mem row_d, col_d, max_d, color_d, node_value_d, stop_d1, stop_d2;
	row_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));
	col_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_edges * sizeof(cl_int));
	stop_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	s_array_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));
	node_value_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_float));
	min_array_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_float));
	c_array_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));
	c_u_array_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));


	cl_int zero = 0;
	cl_int n_one = -1;
	
	// Just a flag value
	cl_float max_init = 6666666.0;
	int err = exec->exec_queue.enqueueWriteBuffer(s_array_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, zeros);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(c_array_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), n_ones);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(c_u_array_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), n_ones);
	check_ocl(err);
	err = exec->exec_queue.enqueueFillBuffer(min_array_d, &max_init, 0, num_nodes * sizeof(cl_int));
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(node_value_d, CL_TRUE, 0, num_nodes * sizeof(cl_float), node_value);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(row_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), csr->row_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array);
	check_ocl(err);
}

void reset_persistent_task(CL_Execution *exec) {
	cl_int zero = 0;
	cl_int n_one = -1;
	
	// Just a flag value
	cl_float max_init = 6666666.0;
	
	int err = exec->exec_queue.enqueueWriteBuffer(s_array_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, zeros);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(c_array_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), n_ones);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(c_u_array_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), n_ones);
	check_ocl(err);
	err = exec->exec_queue.enqueueFillBuffer(min_array_d, &max_init,  0, num_nodes * sizeof(cl_int));
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(node_value_d, CL_TRUE, 0, num_nodes * sizeof(cl_float), node_value);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(row_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), csr->row_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array);

}

int set_persistent_app_args_for_occupancy(int arg_index, cl::Kernel k) {
	// Set args for persistent kernel
	int err = k.setArg(arg_index, row_d);
	arg_index++;
	err |= k.setArg(arg_index, col_d);
	arg_index++;
	err = k.setArg(arg_index, node_value_d);
	arg_index++;
	err |= k.setArg(arg_index, s_array_d);
	arg_index++;
	err = k.setArg(arg_index, c_array_d);
	arg_index++;
	err |= k.setArg(arg_index, c_u_array_d);
	arg_index++;
	err = k.setArg(arg_index, min_array_d);
	arg_index++;
	err |= k.setArg(arg_index, stop_d);
	arg_index++;
	err |= k.setArg(arg_index, num_nodes);
	arg_index++;
	err |= k.setArg(arg_index, num_edges);
	arg_index++;
	check_ocl(err);

	return arg_index;
}

// This is pannotia specific
void output_persistent_solution(const char *fname, CL_Execution *exec) {

	exec->exec_queue.enqueueReadBuffer(s_array_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, s_array_output);
	FILE * fp = fopen(fname, "w");
	if (!fp) { printf("ERROR: unable to open file %s\n", FLAGS_graph_output.c_str()); }

	for (int i = 0; i < num_nodes; i++)
		fprintf(fp, "%d: %d\n", i + 1, s_array_output[i]);

	fclose(fp);
}

void clean_persistent_task(CL_Execution *exec) {
	free(s_array_output);
	free(node_value);
	free(zeros);
	free(n_ones);
}

// This is pannotia specific
bool diff_solution_file_int(int * a, const char * solution_fname, int v) {
	bool ret = true;
	FILE * fp = fopen(solution_fname, "r");
	for (int i = 0; i < v; i++) {
		if (feof(fp)) {
			//printf("111\n");
			ret = false;
			break;
		}
		int compare;
		int trash;
		int found = fscanf(fp, "%d\n", &compare);
		if (found != 1) {
			//printf("222\n");
			ret = false;
			break;
		}
		if (compare != a[i]) {
			//printf("%d found %d expected %d\n", i, compare, a[i]);
			ret = false;
			break;
		}
	}

	fclose(fp);
	return ret;
}

// return true if correct, false otherwise
bool check_persistent_task(CL_Execution *exec) {
	exec->exec_queue.enqueueReadBuffer(s_array_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, s_array_output);
	return diff_solution_file_int(s_array_output, FLAGS_graph_solution_file.c_str(), num_nodes);
}
