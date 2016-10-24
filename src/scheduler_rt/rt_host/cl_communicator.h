#pragma once

#include "cl_execution.h"
#include "cl_scheduler.h"
#include <CL/cl2.hpp>
#include <assert.h>
#include <thread>
#include <chrono>
#include "discovery.h"
#include <vector>

typedef uint64_t time_stamp;
typedef std::pair<time_stamp, time_stamp> time_ret;
typedef std::pair<time_stamp, int> time_groups;
typedef std::tuple<time_stamp, time_stamp, time_stamp> triple_time;


class CL_Communicator {
	
	private:
		CL_Execution *exec;
		std::string mega_kernel_name;
		CL_Scheduler_ctx scheduler;
		int participating_groups;
		std::atomic<int> executing_persistent;
		std::atomic<int> waiting_for_async;
		time_stamp persistent_begin, persistent_end;
		time_stamp global_start;
		Discovery_ctx d_ctx_host;
		cl::Buffer *d_ctx_device;
		std::vector<time_groups> time_groups_data;
		std::vector<triple_time> response_exec_data;
		bool record_time_groups_data;
	public:

		void my_yield() {
			std::this_thread::yield();
		}

		CL_Communicator(CL_Execution &exec_arg, std::string mk_name, CL_Scheduler_ctx s_ctx, cl::Buffer *d_ctx_mem) {
			exec = &exec_arg;
			mega_kernel_name = mk_name;
			scheduler = s_ctx;
			executing_persistent = 0;
			waiting_for_async = 0;
			persistent_begin = persistent_end = 0;
            participating_groups = 0;
			mk_init_discovery_ctx(&d_ctx_host);
			d_ctx_device = d_ctx_mem;
			int err = exec->exec_queue.enqueueWriteBuffer(*d_ctx_device, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx_host);
			check_ocl(err);
			global_start = 0;
			record_time_groups_data = false;
		}

		int launch_mega_kernel(cl::NDRange global_size, cl::NDRange local_size) {
			std::flush(std::cout);

			// Make sure everything is finished. Seems to help with AMD
			int err = exec->exec_queue.flush();
			check_ocl(err);
			err = exec->exec_queue.finish();
			check_ocl(err);

			err = exec->exec_queue.enqueueNDRangeKernel(exec->exec_kernels[mega_kernel_name],
			cl::NullRange,
			global_size,
			local_size);
			check_ocl(err);

			// Tyler: required for AMD, otherwise SVM doesn't seem to work.
			err = exec->exec_queue.flush();
			check_ocl(err);

			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_acquire) != DEVICE_WAITING);
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

		uint64_t gettime_chrono() {
			//return 0;
			return std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::high_resolution_clock::now().time_since_epoch())
				.count();
		}

		// Should possibly check groups here again to make sure we don't ask for too many (compare to participating groups)
		time_ret send_task_synchronous(int groups, const char * label) {

			if (groups > participating_groups) {
				send_quit_signal();
				std::cout << "WARNING CL_Communicator::task_synchronous(): cannot send persistent task with " << groups << " workgroups since there are only " << participating_groups << " participating groups" << std::endl;
				std::flush(std::cout);
				exit(EXIT_FAILURE);
			}

			// For timing. Should be better engineered.
			unsigned long long response_begin, response_end, execution_begin, execution_end;

			*(scheduler.task_size) = groups;
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_TASK, std::memory_order_release);
			std::string buf(label);
			buf.append("_response");

			response_begin = gettime_chrono();
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_relaxed) != DEVICE_GOT_GROUPS) {
				my_yield();
			}
			std::atomic_thread_fence(std::memory_order_acquire);
			response_end = gettime_chrono();

			std::string buf2(label);
			buf.append("_execution");

			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_EXECUTE, std::memory_order_release);
			
			execution_begin = gettime_chrono(); //start application timer here
			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_relaxed) != DEVICE_WAITING) {
				my_yield();
			}
			std::atomic_thread_fence(std::memory_order_acquire);
			execution_end = gettime_chrono(); //end application timer after waiting here.

			time_ret ret = std::make_pair(execution_end - execution_begin, response_end - response_begin);
			triple_time record = std::make_tuple(response_begin, response_end, execution_end);
			response_exec_data.push_back(record);
			return ret;
		}

		void monitor_persistent_task(int groups) {

			*(scheduler.task_size) = groups;
			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_PERSISTENT_TASK, std::memory_order_release);

			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_relaxed) != DEVICE_GOT_GROUPS) {
				my_yield();
			}

			std::atomic_store_explicit((std::atomic<int> *) (scheduler.scheduler_flag), DEVICE_TO_EXECUTE, std::memory_order_release);
			global_start = gettime_chrono();
			std::atomic_store(&executing_persistent, 1);

			persistent_begin = gettime_chrono();

			while (std::atomic_load_explicit((std::atomic<int> *)(scheduler.scheduler_flag), std::memory_order_relaxed) != DEVICE_WAITING) {
				my_yield();
			}

			std::atomic_store(&waiting_for_async, 0);

			while (true) {
				int groups = std::atomic_load_explicit((std::atomic<int> *)(scheduler.persistent_flag), std::memory_order_relaxed);
				if (record_time_groups_data) {
					time_stamp ts = gettime_chrono();
					time_groups_data.push_back(std::make_pair(ts, groups));
				}
				if (groups == 0) {
					break;
				}
				my_sleep(1);
			}

			persistent_end = gettime_chrono();

			std::atomic_store(&executing_persistent, 0);
		}

		void send_persistent_task(int groups) {

			if (groups > participating_groups) {
				send_quit_signal();
				std::cout << "WARNING CL_Communicator::send_persistent_task(): cannot send persistent task with " << groups << " workgroups since there are only " << participating_groups << " participating groups" << std::endl;
				std::flush(std::cout);
				exit(EXIT_FAILURE);
			}

			waiting_for_async = 1;

			std::thread monitor(&CL_Communicator::monitor_persistent_task, this, groups);

			monitor.detach();

			while (atomic_load(&waiting_for_async) != 0)
				;
		}

		int is_executing_persistent() {
			return std::atomic_load(&executing_persistent);
		}

		time_stamp get_persistent_time() {
			assert(std::atomic_load(&executing_persistent) == 0);
			return persistent_end - persistent_begin;
		}

		double reduce_times_ms(std::vector<time_stamp> v) {
			double total = 0.0;
			for (int i = 0; i < v.size(); i++) {
				total += v[i];
			}
			return total / 1000000.0;
		}

		double get_average_time_ms(std::vector<time_stamp> v) {
			if (v.size() == 0) {
				return 0.0;
			}
			double total = reduce_times_ms(v);
		
			return (total / double(v.size()));
		}

		double nano_to_milli(time_stamp t) {
			return double(t) / 1000000.0;
		}

		void my_sleep(int ms) {
			std::this_thread::sleep_for(std::chrono::milliseconds(ms));
		}

		float normalize_ms(time_stamp t) {
			t = t - global_start;
			return nano_to_milli(t);
		}

		void set_record_groups_time_data() {
			record_time_groups_data = true;
		}

		void print_groups_time_data(const char * fname) {
			FILE * fp = fopen(fname, "w");
			if (!fp) { printf("ERROR: unable to open file %s\n", fname); }
			fprintf(fp, "time(ms), groups\n");
			for (int i = 0; i < time_groups_data.size(); i++)
				fprintf(fp, "%f, %d\n", normalize_ms(time_groups_data[i].first), time_groups_data[i].second);

			fclose(fp);
		}

		void print_response_exec_data(const char * fname) {
			FILE * fp = fopen(fname, "w");
			if (!fp) { printf("ERROR: unable to open file %s\n", fname); }
			fprintf(fp, "response_begin, execution_begin, execution_end\n");
			for (int i = 0; i < response_exec_data.size(); i++)
				fprintf(fp, "%f, %f, %f\n", normalize_ms(std::get<0>(response_exec_data[i])),
					                       normalize_ms(std::get<1>(response_exec_data[i])),
					                       normalize_ms(std::get<2>(response_exec_data[i])));

			fclose(fp);
		}

		int get_occupancy_bound(int local_size) {
			launch_mega_kernel(cl::NDRange(MAX_P_GROUPS * local_size), cl::NDRange(local_size));
			int ret = number_of_discovered_groups();
			send_quit_signal();
			int err = exec->exec_queue.flush();
			check_ocl(err);
			err = exec->exec_queue.finish();
			check_ocl(err);
			participating_groups = 0;
			executing_persistent = 0;
			waiting_for_async = 0;
			persistent_begin = persistent_end = 0;
			participating_groups = 0;
			restart_scheduler(&scheduler);
			mk_init_discovery_ctx(&d_ctx_host);
			err = exec->exec_queue.enqueueWriteBuffer(*d_ctx_device, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx_host);
			check_ocl(err);
			global_start = 0;
			return ret;
		}

};
