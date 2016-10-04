#pragma once

#include "cl_execution.h"
#include "cl_scheduler.h"
#include <CL/cl2.hpp>

// needed for timers
#include <Windows.h>

typedef std::pair<unsigned long long, unsigned long long> time_ret;

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
		time_ret send_task_synchronous(int groups) {

			// For timing. Should be better engineered.
			unsigned long long response_begin, response_end, execution_begin, execution_end;

			*(scheduler.task_size) = groups;
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_TASK, std::memory_order_release);

			response_begin = gettime_Windows();
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_acquire) != DEVICE_GOT_GROUPS);
			response_end = gettime_Windows();

			execution_begin = gettime_Windows(); //start application timer here
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_EXECUTE, std::memory_order_release);
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_acquire) != DEVICE_WAITING);
			execution_end = gettime_Windows(); //end application timer after waiting here.
			time_ret ret = std::make_pair(execution_end - execution_begin, response_end - response_begin);
			return ret;
		}

		void send_persistent_task(int groups) {
			*(scheduler.task_size) = groups;
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_PERSISTENT_TASK, std::memory_order_release);
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_acquire) != DEVICE_WAITING);
		}
		
	
};