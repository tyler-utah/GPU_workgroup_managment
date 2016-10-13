
#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include<iostream>

// Should be in a directory somewhere probably. Or defined in CMake.
#define CL_INT_TYPE cl_int
#define ATOMIC_CL_INT_TYPE cl_int
#define MY_CL_GLOBAL 
#define CL_UCHAR_TYPE cl_uchar

#include "base/commandlineflags.h"
#include "opencl/opencl.h"
#include "cl_execution.h"
#include "discovery.h"
#include "cl_communicator.h"
#include "cl_scheduler.h"
#include "kernel_ctx.h"
#include "iw_barrier.h"
#include "base/file.h"

DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_bool(list, false, "List OpenCL platforms and devices");
DEFINE_string(scheduler_rt_path, "uvm_tests/test1/include/rt_device", "Path to scheduler runtime includes");
DEFINE_string(restoration_ctx_path, "tyler_handwritten_tests/first_resize/common/", "Path to restoration context");


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
	cl_float4 x;
	return ret;
}

int main(int argc, char *argv[]) {

	flags::ParseCommandLineFlags(&argc, &argv, true);

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
	err = exec.compile_kernel(kernel_file, file::Path(FLAGS_scheduler_rt_path), file::Path(FLAGS_restoration_ctx_path));

	check_ocl(err);

	get_kernels(exec);

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

	// persistent kernel args
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

	// Set args for persistent kernel
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
	CL_Communicator cl_comm(exec, "mega_kernel", cl::NDRange(wg_size * local_size), cl::NDRange(local_size), s_ctx);

	// Launch the mega kernel
	err = cl_comm.launch_mega_kernel();
	check_ocl(err);

	// Get the number of found groups
	int participating_groups = cl_comm.number_of_discovered_groups();

	cl_comm.send_persistent_task(participating_groups);

	time_ret first_time = cl_comm.send_task_synchronous(1, "first");
	int first_found = *graphics_result;
	*graphics_result = INT_MAX;
	time_ret second_time = cl_comm.send_task_synchronous(1, "second");
	int second_found = *graphics_result;
	*graphics_result = INT_MAX;
	time_ret third_time = cl_comm.send_task_synchronous(1, "third");
	int third_found = *graphics_result;

	cl_comm.send_quit_signal();
	err = exec.exec_queue.finish();
	check_ocl(err);

	cout << "number of participating groups: " << participating_groups << endl;

	cout << "min expected: " << arr_min << " min found: " << first_found << " " <<  second_found << " " << third_found  << endl;
	cout << "time for " << participating_groups << " workgroups: " << cl_comm.nano_to_milli(first_time.second) << " " << cl_comm.nano_to_milli(first_time.first) << " ms" << endl;
	cout << "time for " << participating_groups / 2 << " workgroups: " << cl_comm.nano_to_milli(second_time.second) << " " << cl_comm.nano_to_milli(second_time.first) << " ms" << endl;
	cout << "time for " << participating_groups / 4 << " workgroups: " << cl_comm.nano_to_milli(third_time.second) << " " << cl_comm.nano_to_milli(third_time.first) << " ms" << endl;

	cout << "time for persistent kernel " << cl_comm.nano_to_milli(cl_comm.get_persistent_time()) << endl;

	
	cout << "hello world " << *(s_ctx.participating_groups) << endl;

	free_scheduler_ctx(&exec, &s_ctx);

	//profile::PrintProfileTraceAtResolution(&std::cout, profile::Milliseconds);
	return 0;
}