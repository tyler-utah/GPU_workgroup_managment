
#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include<iostream>
#include <vector>
#include "limits.h"

#include "../rt_common/cl_types.h"
#include "base/commandlineflags.h"
#include "opencl/opencl.h"
#include "cl_execution.h"
#include "discovery.h"
#include "cl_communicator.h"
#include "../common/restoration_ctx.h"
#include "cl_scheduler.h"
#include "kernel_ctx.h"
#include "iw_barrier.h"
#include "base/file.h"

#include "parse.h"

DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_bool(list, false, "List OpenCL platforms and devices");
DEFINE_string(scheduler_rt_path, "scheduler_rt/rt_device", "Path to scheduler runtime includes");
DEFINE_string(restoration_ctx_path, "tyler_handwritten_tests/color_handwritten/common/", "Path to restoration context");
DEFINE_string(graph_file, "", "Path to the graph_file");
DEFINE_string(output, "", "Path to output the result");
DEFINE_int32(non_persistent_wgs, 2, "ratio of workgroups to send to non-persistent task. Special values are (-1) to send all but one workgroup and (-2) to send one workgroup");
DEFINE_int32(skip_tasks, 0, "flag to say if non persistent tasks should be skipped: 0 - don't skip, 1 - skip");
DEFINE_string(kernel_file, "tyler_handwritten_tests/color_handwritten/device/mega_kernel.cl", "the path the mega kernel file");
DEFINE_int32(non_persistent_frequency, 100, "frequency in milliseconds to launch non_persistent tasks");


using namespace std;

int num_nodes = 0, num_edges = 0;
csr_array *csr;
cl_int *color;
cl_float *node_value;
cl::Buffer row_d, col_d, stop_d1, stop_d2, color_d, node_value_d, max_d;
CL_Execution exec;

const char* graph_app_name() {
	return "color";
}

void init_graph_app() {

	csr = parseMetis(FLAGS_graph_file.c_str(), &num_nodes, &num_edges, 0);

	color = (cl_int *)malloc(num_nodes * sizeof(cl_int));
	node_value = (cl_float *)malloc(num_nodes * sizeof(cl_float));
	srand(6);
	for (int i = 0; i < num_nodes; i++) {
		color[i] = -1;

		// Original application: Node_value[i] =  rand()/(float)RAND_MAX;
		node_value[i] = i / (float)(num_nodes + 1);
		//node_value[i] = rand() / (float)RAND_MAX;
	}

	//cl_mem row_d, col_d, max_d, color_d, node_value_d, stop_d1, stop_d2;
	row_d = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));
	col_d = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, num_edges * sizeof(cl_int));
	stop_d1 = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	stop_d2 = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int));
	color_d = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_int));
	node_value_d = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_float));
	max_d = cl::Buffer(exec.exec_context, CL_MEM_READ_WRITE, num_nodes * sizeof(cl_float));

	cl_int zero = 0;
	exec.exec_queue.enqueueWriteBuffer(color_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), color);
	exec.exec_queue.enqueueWriteBuffer(max_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), color);
	exec.exec_queue.enqueueWriteBuffer(node_value_d, CL_TRUE, 0, num_nodes * sizeof(cl_float), node_value);
	exec.exec_queue.enqueueWriteBuffer(stop_d1, CL_TRUE, 0, sizeof(cl_int), &zero);
	exec.exec_queue.enqueueWriteBuffer(stop_d2, CL_TRUE, 0, sizeof(cl_int), &zero);
	exec.exec_queue.enqueueWriteBuffer(row_d, CL_TRUE, 0, num_nodes * sizeof(cl_int), csr->row_array);
	exec.exec_queue.enqueueWriteBuffer(col_d, CL_TRUE, 0, num_edges * sizeof(cl_int), csr->col_array);
}

int set_graph_app_args(int arg_index) {
	// Set args for persistent kernel
	int err = exec.exec_kernels["mega_kernel"].setArg(arg_index, row_d);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, col_d);
	arg_index++;
	err = exec.exec_kernels["mega_kernel"].setArg(arg_index, node_value_d);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, color_d);
	arg_index++;
	err = exec.exec_kernels["mega_kernel"].setArg(arg_index, stop_d1);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, stop_d2);
	arg_index++;
	err = exec.exec_kernels["mega_kernel"].setArg(arg_index, max_d);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, num_nodes);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, num_edges);
	arg_index++;
	check_ocl(err);

	return arg_index;
}

void output_graph_solution(const char *fname) {

	exec.exec_queue.enqueueReadBuffer(color_d, CL_TRUE, 0, sizeof(cl_int) * num_nodes, color);
	FILE * fp = fopen(fname, "w");
	if (!fp) { printf("ERROR: unable to open file %s\n", FLAGS_output.c_str()); }

	for (int i = 0; i < num_nodes; i++)
		fprintf(fp, "%d: %d\n", i + 1, color[i]);

	fclose(fp);
}

void graph_app_cleanup() {
	free(color);
	free(node_value);
}

//From IWOCL tutorial (needs attribution)
unsigned getDeviceList(std::vector<std::vector<cl::Device> >& devices)
{
	// Get list of platforms
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	// Enumerate devices
	for (unsigned int i = 0; i < platforms.size(); i++)
	{
		std::vector<cl::Device> plat_devices;
		platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
		devices.push_back(plat_devices);
		//devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
	}

	return devices.size();
}

//From IWOCL tutorial (needs attribution)
void list_devices() {

	std::vector<std::vector<cl::Device> > devices;
	getDeviceList(devices);

	// Print device names
	if (devices.size() == 0) {
		std::cout << "No devices found." << std::endl;
	}
	else {
		std::cout << std::endl;
		std::cout << "Platform,Devices:" << std::endl;
		for (unsigned j = 0; j < devices.size(); j++) {
			for (unsigned i = 0; i < devices[j].size(); i++) {
				std::cout << j << ", " << i << ": " << CL_Execution::getDeviceName(devices[j][i]) << std::endl;
			}
		}
	}
}

int get_app_kernels(CL_Execution &exec) {
	int ret = CL_SUCCESS;
	exec.exec_kernels["mega_kernel"] = cl::Kernel(exec.exec_program, "mega_kernel", &ret);
	return ret;
}

int main(int argc, char *argv[]) {

	flags::ParseCommandLineFlags(&argc, &argv, true);

	int err = 0;

	if (FLAGS_list) {
		list_devices();
		exit(0);
	}

	std::vector<std::vector<cl::Device> > devices;
	getDeviceList(devices);
	if (FLAGS_platform_id < 0 || FLAGS_platform_id >= devices.size()) {
		printf("invalid platform id. Please use the --list option to view platforms and device ids\n");
		exit(0);
	}
	if (FLAGS_device_id < 0 || FLAGS_device_id >= devices[FLAGS_platform_id].size()) {
		printf("invalid device id. Please use the --list option to view platforms and device ids\n");
	}

	exec.exec_device = devices[FLAGS_platform_id][FLAGS_device_id];

	printf("Using GPU: %s\n", exec.getExecDeviceName().c_str());

	printf("Executing appliction: %s\n", graph_app_name());


	cl::Context context(exec.exec_device);
	exec.exec_context = context;
	cl::CommandQueue queue(exec.exec_context);
	exec.exec_queue = queue;

	// Should be built into the cmake file. Haven't thought of how to do this yet.
	err = exec.compile_kernel(file::Path(FLAGS_kernel_file.c_str()), file::Path(FLAGS_scheduler_rt_path), file::Path(FLAGS_restoration_ctx_path));

	check_ocl(err);

	get_app_kernels(exec);

	// set up the discovery protocol
	Discovery_ctx d_ctx;
	mk_init_discovery_ctx(&d_ctx);
	cl::Buffer d_ctx_mem (exec.exec_context, CL_MEM_READ_WRITE, sizeof(Discovery_ctx));
	err = exec.exec_queue.enqueueWriteBuffer(d_ctx_mem, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx);
	check_ocl(err);

	// Reduce kernel args. 
	int graphics_arr_length = 1048576;
	cl_int * h_graphics_buffer = (cl_int *) malloc(sizeof(cl_int) * graphics_arr_length);
	int arr_min = INT_MAX;
	for (int i = 0; i < graphics_arr_length; i++) {
		int loop_int = rand() + 1;
		if (loop_int < arr_min) {
			arr_min = loop_int;
		}
		h_graphics_buffer[i] = loop_int;
	}

	cl::Buffer d_graphics_buffer(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int) * graphics_arr_length);
	err = exec.exec_queue.enqueueWriteBuffer(d_graphics_buffer, CL_TRUE, 0, sizeof(cl_int) * graphics_arr_length, h_graphics_buffer);
	
	cl_int * graphics_result = (cl_int*) clSVMAlloc(exec.exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
	*graphics_result = INT_MAX;


	init_graph_app();

	IW_barrier h_bar;
	for (int i = 0; i < MAX_P_GROUPS; i++) {
		h_bar.barrier_flags[i] = 0;
	}
	h_bar.phase = 0;

	cl::Buffer d_bar(exec.exec_context, CL_MEM_READ_WRITE, sizeof(IW_barrier));
	err = exec.exec_queue.enqueueWriteBuffer(d_bar, CL_TRUE, 0, sizeof(IW_barrier), &h_bar);
	check_ocl(err);

	// kernel contexts for the graphics kernel and persistent kernel
	cl::Buffer d_graphics_kernel_ctx(exec.exec_context, CL_MEM_READ_WRITE, sizeof(Kernel_ctx));
	cl::Buffer d_persistent_kernel_ctx(exec.exec_context, CL_MEM_READ_WRITE, sizeof(Kernel_ctx));

	// scheduler context
	CL_Scheduler_ctx s_ctx;
	mk_init_scheduler_ctx(&exec, &s_ctx);

	// Setting the args
	int arg_index = 0;

	// Set the args for graphics kernel
	err = exec.exec_kernels["mega_kernel"].setArg(arg_index, graphics_arr_length);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, d_graphics_buffer);
	arg_index++;
	err |= clSetKernelArgSVMPointer(exec.exec_kernels["mega_kernel"](), arg_index, graphics_result);
	arg_index++;
	check_ocl(err);

	arg_index = set_graph_app_args(arg_index);

	// Set barrier arg
	err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_bar);
	arg_index++;
	check_ocl(err);

	// Set arg for discovery protocol
	err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_ctx_mem);
	arg_index++;
	check_ocl(err);

	// Set arg for kernel contexts
	err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_graphics_kernel_ctx);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, d_persistent_kernel_ctx);
	arg_index++;
	check_ocl(err);

	err = set_scheduler_args(&exec.exec_kernels["mega_kernel"], &s_ctx, arg_index);
	//check_ocl(err);

	// Set up the communicator
	int local_size = 256;
	int wg_size = MAX_P_GROUPS;
	CL_Communicator cl_comm(exec, "mega_kernel", s_ctx, &d_ctx_mem);

	// Launch the mega kernel

	int occupancy_bound = cl_comm.get_occupancy_bound(local_size);

	cl_comm.set_record_groups_time_data();

	
	std::vector<time_stamp> response_time;
	std::vector<time_stamp> execution_time;
	int error = 0;

	int workgroups_for_non_persistent = 0;

	
	err = exec.exec_queue.flush();
	check_ocl(err);
	exec.exec_queue.finish();
	check_ocl(err);
	std::cout << "launching mega kernel..." << std::endl;
	err = cl_comm.launch_mega_kernel(cl::NDRange(local_size * occupancy_bound), cl::NDRange(local_size));

	// Get the number of found groups
	int participating_groups = cl_comm.number_of_discovered_groups();

	if (FLAGS_non_persistent_wgs == -1) {
		workgroups_for_non_persistent = participating_groups - 1;
	}
	else if (FLAGS_non_persistent_wgs == -2) {
		workgroups_for_non_persistent = 1;
	}
	else {
		workgroups_for_non_persistent = participating_groups / FLAGS_non_persistent_wgs;
	}
	

	cl_comm.send_persistent_task(participating_groups);

	// Only do these tasks if specified by a flag
	if (FLAGS_skip_tasks == 0) {
		time_stamp begin = cl_comm.gettime_chrono();
		time_stamp end;
		while (true) {

			// Break if the persistent task isn't executing
			if (!cl_comm.is_executing_persistent()) {
				break;
			}
			end = cl_comm.gettime_chrono();
			
			// Did we redline?
			if (cl_comm.nano_to_milli(end - begin) >= FLAGS_non_persistent_frequency) {
				cl_comm.add_redline();
			}

			// no we didn't redline
			else {

				// Spin until its time to launch the task
				while (true) {
					end = cl_comm.gettime_chrono();

					// Could sleep here if the difference is large!
					if (cl_comm.nano_to_milli(end - begin) >= FLAGS_non_persistent_frequency) {
						break;
					}
				}
			}

			// Reset non-persistent task
			*graphics_result = INT_MAX;

			// Launch the task
			time_ret timing_info = cl_comm.send_task_synchronous(workgroups_for_non_persistent, "first");

			// check the result
			int g_result = *graphics_result;
			if (g_result != 1) {
				error = 1;
			}
			begin = end;
		}
	}

	cl_comm.send_quit_signal();
	err = exec.exec_queue.finish();
	check_ocl(err);

	cout << endl << "error: " << error << endl;

	cout << "check value is: " << *(s_ctx.check_value) << " ms" << endl;


	if (strcmp("", FLAGS_output.c_str()) != 0) {
		cout << "outputing solution to " << FLAGS_output << endl;
		output_graph_solution(FLAGS_output.c_str());
	}

	cl_comm.print_groups_time_data("tmp.txt");

	cl_comm.print_response_exec_data("tmp2.txt");

	cl_comm.print_response_and_execution_times("tmp3.txt");

	cl_comm.print_summary_file("tmp4.txt");

	cl_comm.print_summary();

	free_scheduler_ctx(&exec, &s_ctx);
	graph_app_cleanup();

	return 0;
}
