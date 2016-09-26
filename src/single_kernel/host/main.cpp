
// Should be in a directory somewhere probably. Or defined in CMake.
#define INT_TYPE cl_int;
#define ATOMIC_INT_TYPE cl_int;

#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include<iostream>

#include "base/commandlineflags.h"
#include "opencl/opencl.h"
#include "cl_execution.h"

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
	err = exec.compile_kernel(kernel_file, "C:/Users/Tyler/Documents/GitHub/GPU_workgroup_managment/UVM_test/test1/include/rt_device/");
	check_ocl(err);

	get_kernels(exec);

	cout << "hello world" << endl;
	return 0;
}