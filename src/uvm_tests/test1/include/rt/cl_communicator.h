#pragma once

#include "cl_execution.h"
#include "cl_scheduler.h"
#include <CL/cl2.hpp>

// needed for timers
#include <Windows.h>

class CL_Communicator {
	
	private:
		CL_Execution *exec;
		std::string mega_kernel_name;
		cl::NDRange global_size;
		cl::NDRange local_size;
		CL_Scheduler_ctx scheduler;
		int participating_groups;
	public:

		CL_Communicator(CL_Execution &exec_arg, std::string mk_name, cl::NDRange gs, cl::NDRange ls, CL_Scheduler_ctx s_ctx) {
			exec = &exec_arg;
			mega_kernel_name = mk_name;
			global_size = gs;
			local_size = ls;
			scheduler = s_ctx;
		}

		int launch_mega_kernel() {
			int err = exec->exec_queue.enqueueNDRangeKernel(exec->exec_kernels[mega_kernel_name],
			cl::NullRange,
			global_size,
			local_size);

			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_acquire) != DEVICE_WAITING);
			participating_groups = *(scheduler.participating_groups);
			
			return err;
		}
		
		int number_of_discovered_groups() {
			return participating_groups;
		}

		void send_quit_signal() {
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_QUIT, std::memory_order_release);
		}

		unsigned long long gettime_Windows() {
			SYSTEMTIME time;
			GetSystemTime(&time);
			unsigned long long ret = (time.wDay * 60 * 60 * 1000 * 24) + (time.wHour * 60 * 60 * 1000) + (time.wMinute * 60 * 1000) + (time.wSecond * 1000) + time.wMilliseconds;
			return ret;
		}

		// Should possibly check groups here again to make sure we don't ask for too many (compare to participating groups)
		unsigned long long send_task_synchronous(int groups) {

			// For timing. Should be better engineered.
			unsigned long long begin, end, last;

			*(scheduler.task_size) = groups;
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_TASK, std::memory_order_release);

			//start response timer here
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_acquire) != DEVICE_GOT_GROUPS);
			//end response timer here.

			begin = gettime_Windows(); //start application timer here
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_EXECUTE, std::memory_order_release);
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_acquire) != DEVICE_WAITING);
			end = gettime_Windows(); //end application timer after waiting here.
			last = end - begin;
			return last;
		}
		
	
};