#include <atomic>
#include <iostream>
#include <fstream>
#include <string>

#include "base/commandlineflags.h"
#include "opencl/opencl.h"
#include "cl_execution.h"

DEFINE_string(input, "", "Input path");
DEFINE_string(output, "", "Output path");
DEFINE_int32(platform_id, -1, "OpenCL platform ID to use");
DEFINE_int32(device_id, -1, "OpenCL device ID to use");
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

#define GPU_WAIT 0
#define GPU_ADD 1
#define GPU_MULT 2
#define GPU_QUIT 3

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

	err = exec.compile_kernel(kernel_file, "");

	check_ocl(err);

	get_kernels(exec);
	
	//Tyler: mixing C++ and C opencl is usually pretty frowned on. However, I cannot
	//figure out the C++ api for fine-grained SVM! And there are very few examples.
	cl_int * flag = (cl_int*) clSVMAlloc(exec.exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(cl_int), 4);
	cl_int * data1 = (cl_int*) clSVMAlloc(exec.exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
	cl_int * data2 = (cl_int*)clSVMAlloc(exec.exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
	cl_int * result = (cl_int*)clSVMAlloc(exec.exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
	*flag = GPU_WAIT;
	*data1 = 0;
	*data2 = 0;
	*result = 0;

	err = clSetKernelArgSVMPointer(exec.exec_kernels["mega_kernel"](), 0, flag);
	err |= clSetKernelArgSVMPointer(exec.exec_kernels["mega_kernel"](), 1, data1);
	err |= clSetKernelArgSVMPointer(exec.exec_kernels["mega_kernel"](), 2, data2);
	err |= clSetKernelArgSVMPointer(exec.exec_kernels["mega_kernel"](), 3, result);

	check_ocl(err);
	cout << "got file " << FLAGS_input << endl;
	ifstream infile(FLAGS_input);

	err = exec.exec_queue.enqueueNDRangeKernel(exec.exec_kernels["mega_kernel"],
		cl::NullRange,
		cl::NDRange(1),
		cl::NDRange(1));
	check_ocl(err);
	err = exec.exec_queue.flush();
	check_ocl(err);

	string tag;
	int a, b;
	vector<int> results;
	while (infile >> tag >> a >> b) {
		if (tag.compare("ADD") == 0) {
			*data1 = a;
			*data2 = b;
			std::atomic_store_explicit((std::atomic<int> *) flag, GPU_ADD, std::memory_order_release);
			while (std::atomic_load_explicit((std::atomic<int> *) flag, std::memory_order_acquire) != GPU_WAIT);
			results.push_back(*result);
			//cout << a + b << endl;
		}
		if (tag.compare("MULT") == 0) {
			*data1 = a;
			*data2 = b;
			std::atomic_store_explicit((std::atomic<int> *) flag, GPU_MULT, std::memory_order_release);
			while (std::atomic_load_explicit((std::atomic<int> *) flag, std::memory_order_acquire) != GPU_WAIT);
			results.push_back(*result);
			//cout << a * b << endl;
		}
	}

	std::atomic_store_explicit((std::atomic<int> *) flag, GPU_QUIT, std::memory_order_release);
	err = exec.exec_queue.finish();
	check_ocl(err);

	for (int i = 0; i < results.size(); i++) {
		cout << results[i] << endl;
	}

	return 0;
}
