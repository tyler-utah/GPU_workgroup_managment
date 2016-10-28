#include "parse.h"

// Could probably go into a pannotia header
DEFINE_string(graph_file, "", "Path to the graph_file");
DEFINE_string(graph_output, "", "Path to output the graph result");
DEFINE_string(graph_solution_file, "pannotia/graph_apps/color/device/color_adapted.cl", "the path the mega kernel file");


// ---
DEFINE_string(restoration_ctx_path, "pannotia/graph_apps/color", "Path to restoration context");
DEFINE_string(merged_kernel_file, "pannotia/graph_apps/color/device/merged.cl", "the path the mega kernel file");
DEFINE_string(persistent_kernel_file, "pannotia/graph_apps/color/device/standalone.cl", "the path the mega kernel file");



int num_nodes = 0, num_edges = 0;
csr_array *csr;
cl_int *color_input;
cl_int *color_output;
cl_float *node_value;
cl::Buffer row_d, col_d, stop_d1, stop_d2, color_d, node_value_d, max_d;

const char* persistent_app_name() {
	return "color";
}

const char* persistent_kernel_name() {
	return "color_combined";
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

	color_input = (cl_int *)malloc(num_nodes * sizeof(cl_int));
	color_output = (cl_int *)malloc(num_nodes * sizeof(cl_int));
	node_value = (cl_float *)malloc(num_nodes * sizeof(cl_float));
	srand(6);
	for (int i = 0; i < num_nodes; i++) {
		color_input[i] = -1;

		// Original application: Node_value[i] =  rand()/(float)RAND_MAX;
		node_value[i] = i / (float)(num_nodes + 1);
		//node_value[i] = rand() / (float)RAND_MAX;
	}

	//cl_mem row_d, col_d, max_d, color_d, node_value_d, stop_d1, stop_d2;
	row_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));
	col_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_edges * sizeof(cl_int));
	stop_d1 = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	stop_d2 = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	color_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));
	node_value_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_float));
	max_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_float));

	cl_int zero = 0;
	int err = exec->exec_queue.enqueueWriteBuffer(color_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), color_input);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(max_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), color_input);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(node_value_d, CL_TRUE, 0, num_nodes * sizeof(cl_float), node_value);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop_d1, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop_d2, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(row_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), csr->row_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array);
	check_ocl(err);
}

void reset_persistent_task(CL_Execution *exec) {
	cl_int zero = 0;
	int err = exec->exec_queue.enqueueWriteBuffer(color_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), color_input);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(max_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), color_input);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(node_value_d, CL_TRUE, 0, num_nodes * sizeof(cl_float), node_value);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop_d1, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop_d2, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(row_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), csr->row_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array);
	check_ocl(err);

	for (int i = 0; i < num_nodes; i++) {
		color_output[i] = -1;
	}

}

int set_persistent_app_args_for_occupancy(int arg_index, cl::Kernel k) {
	// Set args for persistent kernel
	int err = k.setArg(arg_index, row_d);
	arg_index++;
	err |= k.setArg(arg_index, col_d);
	arg_index++;
	err = k.setArg(arg_index, node_value_d);
	arg_index++;
	err |= k.setArg(arg_index, color_d);
	arg_index++;
	err = k.setArg(arg_index, stop_d1);
	arg_index++;
	err |= k.setArg(arg_index, stop_d2);
	arg_index++;
	err = k.setArg(arg_index, max_d);
	arg_index++;
	err |= k.setArg(arg_index, num_nodes);
	arg_index++;
	err |= k.setArg(arg_index, num_edges);
	arg_index++;
	check_ocl(err);

	return arg_index;
}

void output_persistent_solution(const char *fname, CL_Execution *exec) {

	exec->exec_queue.enqueueReadBuffer(color_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, color_output);
	FILE * fp = fopen(fname, "w");
	if (!fp) { printf("ERROR: unable to open file %s\n", FLAGS_graph_output.c_str()); }

	for (int i = 0; i < num_nodes; i++)
		fprintf(fp, "%d: %d\n", i + 1, color_output[i]);

	fclose(fp);
}

void clean_persistent_task(CL_Execution *exec) {
	free(color_input);
	free(node_value);
}

bool diff_solution_file_int(int * a, const char * solution_fname, int v) {
	bool ret = true;
	FILE * fp = fopen(solution_fname, "r");
	for (int i = 0; i < v; i++) {
		if (feof(fp)) {
			printf("111\n");
			ret = false;
			break;
		}
		int compare;
		int trash;
		int found = fscanf(fp, "%d: %d\n", &trash, &compare);
		if (found != 2) {
			ret = false;
			break;
		}
		if (compare != a[i]) {
			ret = false;
			break;
		}
	}

	fclose(fp);
	return ret;
}

bool check_persistent_task(CL_Execution *exec) {
	exec->exec_queue.enqueueReadBuffer(color_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, color_output);
	return diff_solution_file_int(color_output, FLAGS_graph_solution_file.c_str(), num_nodes);
}
