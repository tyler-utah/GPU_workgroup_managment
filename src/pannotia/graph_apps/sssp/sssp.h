#include "parse.h"

#define BIG_NUM 99999999

// Could probably go into a pannotia header
DEFINE_string(graph_file, "", "Path to the graph_file");
DEFINE_string(graph_output, "", "Path to output the graph result");
DEFINE_string(graph_solution_file, "pannotia/solutions/sssp_usa.txt", "the path the mega kernel file");


// ---
DEFINE_string(restoration_ctx_path, "pannotia/graph_apps/sssp", "Path to restoration context");
DEFINE_string(merged_kernel_file, "pannotia/graph_apps/sssp/device/merged.cl", "the path the mega kernel file");
DEFINE_string(persistent_kernel_file, "pannotia/graph_apps/sssp/device/standalone.cl", "the path the mega kernel file");

int num_nodes = 0, num_edges = 0;
csr_array *csr;
cl_int *cost_array_input;
cl_int *cost_array_output;
cl::Buffer row_d, col_d, stop_d, data_d, x_d, y_d;
int source_vertex = 0;

const char* persistent_app_name() {
	return "sssp";
}

const char* persistent_kernel_name() {
	return "sssp_combined";
}

// Empty for Pannotia apps
void init_persistent_app_for_real(CL_Execution *exec, int occupancy) {
	return;
}

void set_persistent_app_args_for_real(int arg_index, cl::Kernel k) {
	return;
}


void init_persistent_app_for_occupancy(CL_Execution *exec) {

	csr = parseCOO_transpose(FLAGS_graph_file.c_str(), &num_nodes, &num_edges, 1);

	cost_array_input = (cl_int *)malloc(num_nodes * sizeof(cl_int));
	cost_array_output = (cl_int *)malloc(num_nodes * sizeof(cl_int));


	for (int i = 0; i < num_nodes; i++) {
		cost_array_output[i] = 0;
		if (i == source_vertex) {
			cost_array_input[i] = 0;
		}
		else {
			cost_array_input[i] = BIG_NUM;
		}
	}

	//Tyler: watch out for the + 1 in row_d! Its a pain to debug if you forget >:(
	row_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_nodes + 1) * sizeof(cl_int));
	col_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_edges * sizeof(cl_int));
	stop_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	data_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_edges * sizeof(cl_int));
	x_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));
	y_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));

	cl_int zero = 0;
	int err = 0;
	err = exec->exec_queue.enqueueWriteBuffer(row_d, CL_TRUE, 0, (num_nodes + 1) * sizeof(cl_int), csr->row_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(data_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->data_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(x_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), cost_array_input);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(y_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), cost_array_input);
	check_ocl(err);
}

void reset_persistent_task(CL_Execution *exec) {
	cl_int zero = 0;
	int err = 0;
	err = exec->exec_queue.enqueueWriteBuffer(row_d, CL_TRUE, 0, (num_nodes + 1)  * sizeof(cl_int), csr->row_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(data_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->data_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(x_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), cost_array_input);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(y_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), cost_array_input);
	check_ocl(err);
	
	
	for (int i = 0; i < num_nodes; i++) {
		cost_array_output[i] = 0;
	}
	
}

int set_persistent_app_args_for_occupancy(int arg_index, cl::Kernel k) {
	// Set args for persistent kernel
	int err = k.setArg(arg_index, num_nodes);
	arg_index++;
	err |= k.setArg(arg_index, row_d);
	arg_index++;
	err |= k.setArg(arg_index, col_d);
	arg_index++;
	err |= k.setArg(arg_index, data_d);
	arg_index++;
	err |= k.setArg(arg_index, x_d);
	arg_index++;
	err |= k.setArg(arg_index, y_d);
	arg_index++;
	err |= k.setArg(arg_index, stop_d);
	arg_index++;
	check_ocl(err);

	return arg_index;
}

void output_persistent_solution(const char *fname, CL_Execution *exec) {

	exec->exec_queue.enqueueReadBuffer(x_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, cost_array_output);
	FILE * fp = fopen(fname, "w");
	if (!fp) { printf("ERROR: unable to open file %s\n", FLAGS_graph_output.c_str()); }

	for (int i = 0; i < num_nodes; i++)
		fprintf(fp, "%d: %d\n", i + 1, cost_array_output[i]);

	fclose(fp);
}

void clean_persistent_task(CL_Execution *exec) {
	free(cost_array_input);
	free(cost_array_output);
}

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
		int found = fscanf(fp, "%d: %d\n", &trash, &compare);
		if (found != 2) {
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

bool check_persistent_task(CL_Execution *exec) {
	//output_persistent_solution("testing.txt", exec);
	exec->exec_queue.enqueueReadBuffer(x_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, cost_array_output);
	return diff_solution_file_int(cost_array_output, FLAGS_graph_solution_file.c_str(), num_nodes);
}
