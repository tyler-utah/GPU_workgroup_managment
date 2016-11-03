#include "bc_parse.h"

// Could probably go into a pannotia header
DEFINE_string(graph_file, "", "Path to the graph_file");
DEFINE_string(graph_output, "", "Path to output the graph result");
DEFINE_string(graph_solution_file, "pannotia/solutions/sssp_usa.txt", "the path the mega kernel file");


// ---
DEFINE_string(restoration_ctx_path, "pannotia/graph_apps/bc", "Path to restoration context");
DEFINE_string(merged_kernel_file, "pannotia/graph_apps/bc/device/merged.cl", "the path the mega kernel file");
DEFINE_string(persistent_kernel_file, "pannotia/graph_apps/bc/device/standalone.cl", "the path the mega kernel file");

cl::Buffer row_d, col_d, row_trans_d, col_trans_d, dist_d, rho_d, sigma_d, p_d, stop1_d, stop2_d, stop3_d, global_dist_d, bc_d;
cl_float *bc_input_h, *bc_output_h;
int num_nodes, num_edges;
csr_array *csr;

const char* persistent_app_name() {
	return "bc";
}

const char* persistent_kernel_name() {
	return "bc_combined";
}

// Empty for Pannotia apps
void init_persistent_app_for_real(CL_Execution *exec, int occupancy) {
	return;
}

void set_persistent_app_args_for_real(int arg_index, cl::Kernel k) {
	return;
}

void init_persistent_app_for_occupancy(CL_Execution *exec) {
	csr = parseCOO(FLAGS_graph_file.c_str(), &num_nodes, &num_edges, 1);
	bc_input_h = (cl_float *) malloc(sizeof(cl_float) * num_nodes);
	bc_output_h = (cl_float *) malloc(sizeof(cl_float) * num_nodes);
	
	for (int i = 0; i < num_nodes; i++) {
		bc_input_h[i] = 0;
		bc_output_h[i] = 0;
	}

	
	bc_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_nodes) * sizeof(cl_float));
	dist_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_nodes) * sizeof(cl_int));
	sigma_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_nodes) * sizeof(cl_float));
    rho_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_nodes) * sizeof(cl_float));
	p_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_nodes) * (num_nodes) * sizeof(cl_int));
	stop1_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	stop2_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	stop3_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	global_dist_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	row_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_nodes + 1) * sizeof(cl_int));
	col_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_edges) * sizeof(cl_int));
	row_trans_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_nodes + 1) * sizeof(cl_int));
	col_trans_d = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, (num_edges) * sizeof(cl_int));
	
	cl_int zero = 0;
	int err = 0;
	err = exec->exec_queue.enqueueWriteBuffer(row_d, CL_TRUE, 0, (num_nodes + 1) * sizeof(cl_int), csr->row_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(row_trans_d, CL_TRUE, 0, (num_nodes + 1) * sizeof(cl_int), csr->row_array_t);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_trans_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array_t);
	check_ocl(err);
	
	err = exec->exec_queue.enqueueWriteBuffer(stop1_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop2_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop3_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	
	err = exec->exec_queue.enqueueWriteBuffer(bc_d, CL_TRUE, 0, sizeof(cl_float) * num_nodes, bc_input_h);
	check_ocl(err);
	
}

void reset_persistent_task(CL_Execution *exec) {
	cl_int zero = 0;
	int err = 0;
	err = exec->exec_queue.enqueueWriteBuffer(row_d, CL_TRUE, 0, (num_nodes + 1) * sizeof(cl_int), csr->row_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(row_trans_d, CL_TRUE, 0, (num_nodes + 1) * sizeof(cl_int), csr->row_array_t);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(col_trans_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array_t);
	check_ocl(err);
	
	err = exec->exec_queue.enqueueWriteBuffer(stop1_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop2_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	err = exec->exec_queue.enqueueWriteBuffer(stop3_d, CL_TRUE, 0, sizeof(cl_int), &zero);
	check_ocl(err);
	
	err = exec->exec_queue.enqueueWriteBuffer(bc_d, CL_TRUE, 0, sizeof(cl_float) * num_nodes, bc_input_h);
	check_ocl(err);
	
	for (int i = 0; i < num_nodes; i++) {
		bc_input_h[i] = 0;
		bc_output_h[i] = 0;
	}
}

int set_persistent_app_args_for_occupancy(int arg_index, cl::Kernel k) {
	
	// Set args for persistent kernel
	int err = k.setArg(arg_index, row_d); // 0
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, col_d); // 1
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, row_trans_d); // 2
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, col_trans_d); // 3
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, dist_d); // 4
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, rho_d); // 5
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, sigma_d); // 6
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, p_d); // 7
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, stop1_d); // 8
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, stop2_d); // 9
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, stop3_d);  // 10
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, global_dist_d);  // 11
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, bc_d); // 12
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, num_nodes); // 13
	arg_index++;
	check_ocl(err);
	err |= k.setArg(arg_index, num_edges); // 14
	arg_index++;
	check_ocl(err);

	return arg_index;
}

void clean_persistent_task(CL_Execution *exec) {
	free(bc_input_h);
	free(bc_output_h);
}

float my_abs(float a) {
  if (a < 0)
    return -a;
  return a;
}

bool diff_solution_file_int(float * a, const char * solution_fname, int v) {
	bool ret = true;
	FILE * fp = fopen(solution_fname, "r");
	//printf("\n\nSTARTING ANALYSIS\n\n");
	for (int i = 0; i < v; i++) {
		//printf("%f\n", a[i]);
		if (feof(fp)) {
			printf("111\n");
			ret = false;
			break;
		}
		float compare;
		int found = fscanf(fp, "%f\n", &compare);
		if (found != 1) {
			printf("222\n");
			ret = false;
			break;
		}
		if (my_abs(compare - a[i]) > .01) { // .001 as a threshold seems to work.
			printf("ERROR %d found %f expected %f\n", i, compare, a[i]);
			ret = false;
			break;
		}
	}

	fclose(fp);
	return ret;
}

bool check_persistent_task(CL_Execution *exec) {
	//output_persistent_solution("testing.txt", exec);
	exec->exec_queue.enqueueReadBuffer(bc_d, CL_TRUE, 0, sizeof(cl_float) * num_nodes, bc_output_h);
	//for (int i = 0; i < num_nodes; i++) {
	//	printf("%f\n", bc_output_h[i]);
	//}
	return diff_solution_file_int(bc_output_h, FLAGS_graph_solution_file.c_str(), num_nodes);
}
