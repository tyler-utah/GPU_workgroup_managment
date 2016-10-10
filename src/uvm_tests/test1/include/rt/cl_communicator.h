#pragma once

#include "cl_execution.h"
#include "cl_scheduler.h"
#include <CL/cl2.hpp>
#include <assert.h>
#include <thread>

// needed for timers
#include <Windows.h>

typedef unsigned long long time_stamp;
typedef std::pair<time_stamp, time_stamp> time_ret;

class CL_Communicator {
	
	private:
		CL_Execution *exec;
		std::string mega_kernel_name;
		cl::NDRange global_size;
		cl::NDRange local_size;
		CL_Scheduler_ctx scheduler;
		int participating_groups;
		volatile bool executing_persistent;
		volatile time_stamp persistent_begin, persistent_end;
	public:

		CL_Communicator(CL_Execution &exec_arg, std::string mk_name, cl::NDRange gs, cl::NDRange ls, CL_Scheduler_ctx s_ctx) {
			exec = &exec_arg;
			mega_kernel_name = mk_name;
			global_size = gs;
			local_size = ls;
			scheduler = s_ctx;
			executing_persistent = false;
			persistent_begin = persistent_end = 0;
		}

		int launch_mega_kernel() {
			int err = exec->exec_queue.enqueueNDRangeKernel(exec->exec_kernels[mega_kernel_name],
			cl::NullRange,
			global_size,
			local_size);

			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_relaxed) != DEVICE_WAITING);
			std::atomic_thread_fence(std::memory_order_acquire);
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
		time_ret send_task_synchronous(int groups, const char * label) {

			// For timing. Should be better engineered.
			unsigned long long response_begin, response_end, execution_begin, execution_end;

			*(scheduler.task_size) = groups;
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_TASK, std::memory_order_release);
			std::string buf(label);
			buf.append("_response");

			{
				profile::Region region("hi", profile::VERBOSITY_MEDIUM);
				response_begin = gettime_Windows();
				while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_relaxed) != DEVICE_GOT_GROUPS) {
					YieldProcessor();
				}
				std::atomic_thread_fence(std::memory_order_acquire);
				response_end = gettime_Windows();
			}
			std::string buf2(label);
			buf.append("_execution");

			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_EXECUTE, std::memory_order_release);
			
			{
				profile::Region region("hi", profile::VERBOSITY_MEDIUM);
				execution_begin = gettime_Windows(); //start application timer here
				while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_relaxed) != DEVICE_WAITING) {
					YieldProcessor();
				}
				std::atomic_thread_fence(std::memory_order_acquire);
				execution_end = gettime_Windows(); //end application timer after waiting here.
			}
			time_ret ret = std::make_pair(execution_end - execution_begin, response_end - response_begin);
			return ret;
		}

		void monitor_persistent_task() {
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.persistent_flag), std::memory_order_relaxed) != PERSIST_TASK_DONE) {
				YieldProcessor();
			}

			persistent_end = gettime_Windows();

			// Technically a data-race. should be fixed
			executing_persistent = false;
		}

		void send_persistent_task(int groups) {
			*(scheduler.task_size) = groups;
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_PERSISTENT_TASK, std::memory_order_release);
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_relaxed) != DEVICE_GOT_GROUPS) {
				YieldProcessor();
			}

			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_EXECUTE, std::memory_order_release);
			persistent_begin = gettime_Windows();
			executing_persistent = true;
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_relaxed) != DEVICE_WAITING) {
				YieldProcessor();
			}

			std::thread monitor(&CL_Communicator::monitor_persistent_task, this);

			monitor.detach();
		}

		bool is_executing_persistent() {
			return executing_persistent;
		}

		time_stamp get_persistent_time() {
			assert(executing_persistent == false);
			return persistent_end - persistent_begin;
		}
};