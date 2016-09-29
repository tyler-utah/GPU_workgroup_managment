#pragma once

#include "cl_execution.h"
#include <CL/cl2.hpp>

class CL_Communicator {
	
	private:
		CL_Execution *exec;
		std::string mega_kernel_name;
		cl::NDRange global_size;
		cl::NDRange local_size;
		
	public:
		int launch_mega_kernel() {
			int err = exec->exec_queue.enqueueNDRangeKernel(exec->exec_kernels["mega_kernel_name"],
			cl::NullRange,
			cl::NDRange(1),
			cl::NDRange(1));
			
			return err;
		}
		
		int number_of_discovered_groups() {
			return 10;
		}
		
	
};