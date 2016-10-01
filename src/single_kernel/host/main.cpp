
#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include<iostream>

// Should be in a directory somewhere probably. Or defined in CMake.
#define CL_INT_TYPE cl_int
#define ATOMIC_CL_INT_TYPE cl_int
#define MY_CL_GLOBAL 

#include "base/commandlineflags.h"
#include "opencl/opencl.h"
#include "cl_execution.h"
#include "discovery.h"
#include "cl_communicator.h"
#include "cl_scheduler.h"
#include "kernel_ctx.h"

DEFINE_string(input, "", "Input path");
DEFINE_string(output, "", "Output path");
DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_bool(list, false, "List OpenCL platforms and devices");

using namespace std;

const char *kernel_file = XSTR(KERNEL_FILE);

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

int get_kernels(CL_Execution &exec) {
	int ret = CL_SUCCESS;
	exec.exec_kernels["mega_kernel"] = cl::Kernel(exec.exec_program, "mega_kernel", &ret);
	return ret;
}

int main(int argc, char *argv[]) {

	flags::ParseCommandLineFlags(&argc, &argv, true);

	if (FLAGS_input.empty() || FLAGS_output.empty()) {
		printf("Input and output files not specified.");
		exit(0);
	}

	CL_Execution exec;
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

	cl::Context context(exec.exec_device);
	exec.exec_context = context;
	cl::CommandQueue queue(exec.exec_context);
	exec.exec_queue = queue;

	// Should be built into the cmake file. Haven't thought of how to do this yet.
	err = exec.compile_kernel(kernel_file, "C:/Users/Tyler/Documents/GitHub/GPU_workgroup_managment/src/uvm_tests/test1/include/rt_device/");
	check_ocl(err);

	get_kernels(exec);

	Discovery_ctx d_ctx;
	mk_init_discovery_ctx(&d_ctx);
	cl::Buffer d_ctx_mem (exec.exec_context, CL_MEM_READ_WRITE, sizeof(Discovery_ctx));
	err = exec.exec_queue.enqueueWriteBuffer(d_ctx_mem, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx);
	check_ocl(err);

	//Reduce kernel args. 
	int arr_size = 1048576;
	cl_int * h_kernel_buffer = (cl_int *) malloc(sizeof(cl_int) * arr_size);
	int arr_min = INT_MAX;
	for (int i = 0; i < arr_size; i++) {
		int loop_int = rand() + 1;
		if (loop_int < arr_min) {
			arr_min = loop_int;
		}
		h_kernel_buffer[i] = loop_int;
		
	}
	cl::Buffer d_kernel_buffer(exec.exec_context, CL_MEM_READ_WRITE, sizeof(cl_int) * arr_size);

	err = exec.exec_queue.enqueueWriteBuffer(d_kernel_buffer, CL_TRUE, 0, sizeof(cl_int) * arr_size, h_kernel_buffer);


	cl::Buffer d_kernel_ctx(exec.exec_context, CL_MEM_READ_WRITE, sizeof(Kernel_ctx));
	cl_int * kernel_result = (cl_int*) clSVMAlloc(exec.exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
	*kernel_result = INT_MAX;
	CL_Scheduler_ctx s_ctx;
	mk_init_scheduler_ctx(&exec, &s_ctx);

	int arg_index = 0;

	err = exec.exec_kernels["mega_kernel"].setArg(arg_index, d_ctx_mem);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, d_kernel_ctx);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, d_kernel_buffer);
	arg_index++;
	err |= clSetKernelArgSVMPointer(exec.exec_kernels["mega_kernel"](), arg_index, kernel_result);
	arg_index++;
	err |= exec.exec_kernels["mega_kernel"].setArg(arg_index, arr_size);
	arg_index++;

	check_ocl(err);
	int local_size = 256;
	int wg_size = MAX_P_GROUPS;
	CL_Communicator cl_comm(exec, "mega_kernel", cl::NDRange(wg_size * local_size), cl::NDRange(local_size), s_ctx);
	
	err = set_scheduler_args(&exec.exec_kernels["mega_kernel"], &s_ctx, arg_index);
	check_ocl(err);

	err = cl_comm.launch_mega_kernel();
	check_ocl(err);
	int participating_groups = cl_comm.number_of_discovered_groups();

	unsigned long long first_time = cl_comm.send_task_synchronous(participating_groups);
	int first_found = *kernel_result;
	*kernel_result = INT_MAX;
	unsigned long long second_time = cl_comm.send_task_synchronous(participating_groups / 2);
	int second_found = *kernel_result;
	*kernel_result = INT_MAX;
	unsigned long long third_time = cl_comm.send_task_synchronous(participating_groups / 4);
	int third_found = *kernel_result;

	cl_comm.send_quit_signal();
	err = exec.exec_queue.finish();
	check_ocl(err);

	cout << "number of participating groups: " << participating_groups << endl;
	cout << "min expected: " << arr_min << " min found: " << first_found << " " <<  second_found << " " << third_found  << endl;
	cout << "time for " << participating_groups << " workgroups: " << first_time << " ms" << endl;
	cout << "time for " << participating_groups / 2 << " workgroups: " << second_time << " ms" << endl;
	cout << "time for " << participating_groups / 4 << " workgroups: " << third_time << " ms" << endl;


	free_scheduler_ctx(&exec, &s_ctx);
	return 0;
}