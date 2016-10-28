#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <limits.h>

#include "cl_types.h"
#include "base/commandlineflags.h"
#include "opencl/opencl.h"
#include "cl_execution.h"
#include "discovery.h"
#include "cl_communicator.h"
#include "restoration_ctx.h"
#include "cl_scheduler.h"
#include "kernel_ctx.h"
#include "iw_barrier.h"
#include "base/file.h"

DEFINE_int32(platform_id, 0, "OpenCL platform ID to use");
DEFINE_int32(device_id, 0, "OpenCL device ID to use");
DEFINE_bool(list, false, "List OpenCL platforms and devices");
DEFINE_string(scheduler_rt_path, "scheduler_rt/rt_device", "Path to scheduler runtime includes");
DEFINE_string(output_timestamp_executing_groups, "timestamp_executing_groups", "Path to output timestamps and the number of persistent groups");
DEFINE_string(output_timestamp_non_persistent, "timestamp_non_persistent", "Path to output timestamps for non persistent tasks");
DEFINE_string(output_non_persistent_duration, "non_persistent_duration", "Path to output duration results for non persistent tasks");
DEFINE_string(output_summary, "summary", "output file for summary of results");
DEFINE_int32(non_persistent_wgs, 2, "ratio of workgroups to send to non-persistent task. Special values are (-1) to send all but one workgroup and (-2) to send one workgroup");
DEFINE_int32(skip_tasks, 0, "flag to say if non persistent tasks should be skipped: 0 - don't skip, 1 - skip");
DEFINE_int32(non_persistent_frequency, 100, "frequency in milliseconds to launch non_persistent tasks");
DEFINE_int32(is_AMD, 0, "flag to set on AMD chips to reduce occupancy as a work around to the defunct process issue");
DEFINE_int32(use_query_barrier, 0, "flag to use regular ckill barrier or query ckill barrer");



DEFINE_int32(run_non_persistent, 0, "Run only the non persistent task (solo run). If greater than 0, specifies the iterations");
DEFINE_int32(threads_per_wg, 256, "Threads per workgroup for non persistent task solo runs");
DEFINE_int32(num_wgs, 8, "Workgroups for non persistent task solo runs");
DEFINE_int32(merged_iterations, 1, "Iterations to run the merged kernel for");


DEFINE_int32(run_persistent, 0, "Run only the persistent task. If greater than 0, specifies the iterations");


using namespace std;

//Include persistent interface
#if defined PERSISTENT_PANNOTIA_COLOR
#include "../graph_apps/color/color.h"
#elif defined PERSISTENT_OCTREE
#include "host/octree.h"
#else
#error "No persistent task macro defined? like PERSISTENT_XYZ (check your CMakeLists.txt)"
#endif

// Include non-persistent interface
#include "../non_persistent_kernels/reduce/reduce.h"

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

void reset_barrier(CL_Execution *exec, cl::Buffer d_bar) {

	IW_barrier h_bar;
	for (int i = 0; i < MAX_P_GROUPS; i++) {
		h_bar.barrier_flags[i] = 0;
	}
	h_bar.phase = 0;

	int err = exec->exec_queue.enqueueWriteBuffer(d_bar, CL_TRUE, 0, sizeof(IW_barrier), &h_bar);
	check_ocl(err);

}

void reset_discovery(CL_Execution *exec, cl::Buffer d_ctx_mem, bool get_occupancy) {

	Discovery_ctx d_ctx;
	if (get_occupancy) {
		mk_init_discovery_ctx_occupancy(&d_ctx);
	}
	else {
	   mk_init_discovery_ctx(&d_ctx);
	}
	int err = exec->exec_queue.enqueueWriteBuffer(d_ctx_mem, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx);
	check_ocl(err);

}

int discovery_get_occupancy(CL_Execution *exec, cl::Buffer d_ctx_mem) {

	Discovery_ctx d_ctx;
	mk_init_discovery_ctx(&d_ctx);
	int err = exec->exec_queue.enqueueReadBuffer(d_ctx_mem, CL_TRUE, 0, sizeof(Discovery_ctx), &d_ctx);
	check_ocl(err);
	return d_ctx.count;

}

// Just for running the non persistent task
void run_non_persistent(CL_Execution *exec) {
	printf("Running non persistent app %s\n", non_persistent_app_name());
	printf("Running %d iterations\n", FLAGS_run_non_persistent);
	printf("%d threads per workgroup, %d workgroups\n", FLAGS_threads_per_wg, FLAGS_num_wgs);

	int err = exec->compile_kernel(file::Path(FLAGS_non_persistent_kernel_file.c_str()),
		                           file::Path(FLAGS_scheduler_rt_path),
		                           file::Path(FLAGS_restoration_ctx_path),
		                           FLAGS_use_query_barrier);
	check_ocl(err);
	exec->exec_kernels["non_persistent"] = cl::Kernel(exec->exec_program, non_persistent_kernel_name(), &err);
	check_ocl(err);

	init_non_persistent_app(exec);

	int arg_index = 0;
	set_non_persistent_app_args(arg_index, exec->exec_kernels["non_persistent"]);
	err = exec->exec_queue.flush();
	check_ocl(err);
	int error = 0;

	vector<time_stamp> times;

	for (int i = 0; i < FLAGS_run_non_persistent; i++) {
		time_stamp begin = CL_Communicator::gettime_chrono();
		err = exec->exec_queue.enqueueNDRangeKernel(exec->exec_kernels["non_persistent"],
			cl::NullRange,
			cl::NDRange(FLAGS_threads_per_wg * FLAGS_num_wgs),
			cl::NDRange(FLAGS_threads_per_wg));
		check_ocl(err);
		err = exec->exec_queue.finish();
		check_ocl(err);
		time_stamp end = CL_Communicator::gettime_chrono();
		times.push_back(end - begin);

		// check the result
		if (!check_non_persistent_task()) {
			error = 1;
		}

		reset_non_persistent();
	}

	clean_non_persistent_task(exec);

	cout << "Error (should be 0): " << error << endl;
	cout << "Iterations: " << FLAGS_run_non_persistent  << endl;
	cout << "Total time: " << CL_Communicator::reduce_times_ms(times) << " ms" << endl;
	cout << "Mean time: " << CL_Communicator::get_average_time_ms(times) << " ms" << endl;
	cout << "Max time: " << CL_Communicator::max_times_ms(times) << " ms" << endl;
	cout << "Max time: " << CL_Communicator::min_times_ms(times) << " ms" << endl;
	cout << "Standard Deviation: " << CL_Communicator::std_dev_times_ms(times) << " ms" << endl;

}

int get_occupancy_d_ctx(CL_Execution *exec, cl::Kernel k, cl::Buffer d_ctx) {
	int err = exec->exec_queue.flush();
	check_ocl(err);
	err = exec->exec_queue.finish();
	check_ocl(err);
	err = exec->exec_queue.enqueueNDRangeKernel(k,
		cl::NullRange,
		cl::NDRange(FLAGS_threads_per_wg * MAX_P_GROUPS),
		cl::NDRange(FLAGS_threads_per_wg),
		NULL);
	check_ocl(err);
	err = exec->exec_queue.flush();
	check_ocl(err);
	err = exec->exec_queue.finish();
	check_ocl(err);

	return discovery_get_occupancy(exec, d_ctx);
}

int amd_check(int v) {
	if (FLAGS_is_AMD == 1) {
		v = (double)v * .75;
	}
	return v;
}

// Just for running the non persistent task
void run_persistent(CL_Execution *exec) {
	printf("Running non persistent app %s\n", persistent_app_name());
	printf("Running %d iterations\n", FLAGS_run_persistent);

	int err = exec->compile_kernel(file::Path(FLAGS_persistent_kernel_file.c_str()),
		                           file::Path(FLAGS_scheduler_rt_path),
		                           file::Path(FLAGS_restoration_ctx_path),
		                           FLAGS_use_query_barrier);
	check_ocl(err);
	exec->exec_kernels["persistent"] = cl::Kernel(exec->exec_program, persistent_kernel_name(), &err);
	check_ocl(err);

	init_persistent_app_for_occupancy(exec);

	int arg_index = 0;
	int cache_arg_index = arg_index;
	arg_index = set_persistent_app_args_for_occupancy(arg_index, exec->exec_kernels["persistent"]);
	err = exec->exec_queue.flush();
	check_ocl(err);

	cl::Buffer d_bar(exec->exec_context, CL_MEM_READ_WRITE, sizeof(IW_barrier));
	reset_barrier(exec, d_bar);
	err = exec->exec_kernels["persistent"].setArg(arg_index, d_bar);
	arg_index++;
	check_ocl(err);

	cl::Buffer d_ctx_mem(exec->exec_context, CL_MEM_READ_WRITE, sizeof(Discovery_ctx));
	reset_discovery(exec, d_ctx_mem, true);
	err = exec->exec_kernels["persistent"].setArg(arg_index, d_ctx_mem);
	arg_index++;
	check_ocl(err);

	CL_Scheduler_ctx s_ctx;
	mk_init_scheduler_ctx(exec, &s_ctx);
	err = set_scheduler_args(&(exec->exec_kernels["persistent"]), &s_ctx, arg_index);

	int occupancy = get_occupancy_d_ctx(exec, exec->exec_kernels["persistent"], d_ctx_mem);

	int num_wgs = min(FLAGS_num_wgs, amd_check(occupancy));

	printf("%d threads per workgroup, %d workgroups, %d occupancy, %d final size\n", FLAGS_threads_per_wg, FLAGS_num_wgs, occupancy, num_wgs);

	CL_Communicator::my_sleep(1000);

	init_persistent_app_for_real(exec, num_wgs);
	set_persistent_app_args_for_real(cache_arg_index, exec->exec_kernels["persistent"]);

	vector<time_stamp> times;
	vector<int> groups;

	int error = 0;

	for (int i = 0; i < FLAGS_run_persistent; i++) {
		reset_discovery(exec, d_ctx_mem, false);
		reset_barrier(exec, d_bar);
		reset_persistent_task(exec);
		restart_scheduler(&s_ctx);
		CL_Communicator::my_sleep(1000);
		err = exec->exec_queue.flush();
		check_ocl(err);
		err = exec->exec_queue.finish();
		check_ocl(err);

		err = exec->exec_queue.enqueueNDRangeKernel(exec->exec_kernels["persistent"],
			cl::NullRange,
			cl::NDRange(FLAGS_threads_per_wg * num_wgs),
			cl::NDRange(FLAGS_threads_per_wg),
			NULL);
		check_ocl(err);
		err = exec->exec_queue.flush();
		check_ocl(err);
		while (std::atomic_load_explicit((std::atomic<int> *)(s_ctx.scheduler_flag), std::memory_order_acquire) != DEVICE_WAITING);
		time_stamp begin = CL_Communicator::gettime_chrono();
		std::atomic_store_explicit((std::atomic<int> *) (s_ctx.scheduler_flag), DEVICE_TO_PERSISTENT_TASK, std::memory_order_release);
		while (std::atomic_load_explicit((std::atomic<int> *)(s_ctx.persistent_flag), std::memory_order_acquire) != 0);
		time_stamp end = CL_Communicator::gettime_chrono();
		err = exec->exec_queue.finish();
		check_ocl(err);
		//auto elapsed = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		times.push_back(end - begin);

		// check the result
		if (!check_persistent_task(exec)) {
			error = 1;
		}

		int occupancy = discovery_get_occupancy(exec, d_ctx_mem);
		groups.push_back(occupancy);
	}

	//output_persistent_solution(FLAGS_graph_output.c_str(), exec);


	clean_persistent_task(exec);

	cout << endl <<  "Error: " << error << endl << endl;
	cout << "Average occupancy: " << CL_Communicator::average_int_vector(groups) << endl;
	cout << "Iterations: " << FLAGS_run_persistent << endl;
	cout << "Total time: " << CL_Communicator::reduce_times_ms(times) << " ms" << endl;
	cout << "Mean time: " << CL_Communicator::get_average_time_ms(times) << " ms" << endl;
	cout << "Max time: " << CL_Communicator::max_times_ms(times) << " ms" << endl;
	cout << "Min time: " << CL_Communicator::min_times_ms(times) << " ms" << endl;
	cout << "Standard Deviation: " << CL_Communicator::std_dev_times_ms(times) << " ms" << endl;

	cout << "Check value: " << *(s_ctx.check_value) << endl;

}

int get_workgroups_for_non_persistent(int occupancy_bound) {
	int ret;
	if (FLAGS_non_persistent_wgs == -1) {
		ret = occupancy_bound - 1;
	}
	else if (FLAGS_non_persistent_wgs == -2) {
		ret = 1;
	}
	else {
		ret = occupancy_bound / FLAGS_non_persistent_wgs;
	}

	return ret;

}

void execute_merged_iteration(CL_Execution *exec, CL_Communicator &cl_comm, int number_of_workgroups, int workgroups_for_non_persistent, int &error) {

	int err = cl_comm.launch_mega_kernel(cl::NDRange(FLAGS_threads_per_wg * number_of_workgroups), cl::NDRange(FLAGS_threads_per_wg));
	check_ocl(err);

	// Get the number of found groups
	int participating_groups = cl_comm.number_of_discovered_groups();

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
			reset_non_persistent();

			// Launch the task
			cl_comm.send_task_synchronous(workgroups_for_non_persistent, "first");

			// check the result
			if (!check_non_persistent_task()) {
				error = 1;
			}
			begin = end;
		}
	}

	cl_comm.send_quit_signal();
	err = exec->exec_queue.finish();
	check_ocl(err);
}

void run_merged(CL_Execution *exec) {
	printf("Running persistent app %s with non persistent app %s\n", persistent_app_name(), non_persistent_app_name());
	printf("Running %d iterations\n", FLAGS_merged_iterations);
	printf("%d threads per workgroups\n", FLAGS_threads_per_wg);
	printf("Using query barrier: %d\n", FLAGS_use_query_barrier);

	// Should be built into the cmake file. Haven't thought of how to do this yet.
	int err = exec->compile_kernel(file::Path(FLAGS_merged_kernel_file.c_str()),
		                           file::Path(FLAGS_scheduler_rt_path),
		                           file::Path(FLAGS_restoration_ctx_path),
		                           FLAGS_use_query_barrier);

	check_ocl(err);

	exec->exec_kernels["mega_kernel"] = cl::Kernel(exec->exec_program, "mega_kernel", &err);
	check_ocl(err);

	init_persistent_app_for_occupancy(exec);
	init_non_persistent_app(exec);

	int arg_index = 0;
	arg_index = set_non_persistent_app_args(arg_index, exec->exec_kernels["mega_kernel"]);

	int arg_index_cached = arg_index;
	arg_index = set_persistent_app_args_for_occupancy(arg_index, exec->exec_kernels["mega_kernel"]);


	cl::Buffer d_bar(exec->exec_context, CL_MEM_READ_WRITE, sizeof(IW_barrier));
	reset_barrier(exec, d_bar);
	err = exec->exec_kernels["mega_kernel"].setArg(arg_index, d_bar);
	arg_index++;
	check_ocl(err);

	cl::Buffer d_ctx_mem(exec->exec_context, CL_MEM_READ_WRITE, sizeof(Discovery_ctx));
	reset_discovery(exec, d_ctx_mem, true);
	err = exec->exec_kernels["mega_kernel"].setArg(arg_index, d_ctx_mem);
	arg_index++;
	check_ocl(err);

	// kernel contexts for the graphics kernel and persistent kernel
	cl::Buffer d_graphics_kernel_ctx(exec->exec_context, CL_MEM_READ_WRITE, sizeof(Kernel_ctx));
	cl::Buffer d_persistent_kernel_ctx(exec->exec_context, CL_MEM_READ_WRITE, sizeof(Kernel_ctx));

	// Set arg for kernel contexts
	err = exec->exec_kernels["mega_kernel"].setArg(arg_index, d_graphics_kernel_ctx);
	arg_index++;
	err |= exec->exec_kernels["mega_kernel"].setArg(arg_index, d_persistent_kernel_ctx);
	arg_index++;
	check_ocl(err);

	CL_Scheduler_ctx s_ctx;
	mk_init_scheduler_ctx(exec, &s_ctx);
	err = set_scheduler_args(&(exec->exec_kernels["mega_kernel"]), &s_ctx, arg_index);

	// Set up the communicator
	int local_size = FLAGS_threads_per_wg;
	int wg_size = FLAGS_num_wgs;
	CL_Communicator cl_comm(*exec, "mega_kernel", s_ctx, &d_ctx_mem);
	reset_discovery(exec, d_ctx_mem, true);

	int occupancy = get_occupancy_d_ctx(exec, exec->exec_kernels["mega_kernel"], d_ctx_mem);

	int num_wgs = min(FLAGS_num_wgs, amd_check(occupancy));

	printf("%d threads per workgroup, %d workgroups, %d occupancy, %d final size\n", FLAGS_threads_per_wg, FLAGS_num_wgs, occupancy, num_wgs);

	CL_Communicator::my_sleep(1000);

	init_persistent_app_for_real(exec, num_wgs);
	set_persistent_app_args_for_real(arg_index_cached, exec->exec_kernels["mega_kernel"]);

	int error_non_persistent = 0;
	int error_persistent = 0;

	int workgroups_for_non_persistent = get_workgroups_for_non_persistent(num_wgs);

	vector<time_stamp> times;
	cout << endl;
	for (int i = 0; i < FLAGS_merged_iterations; i++) {
		err = exec->exec_queue.flush();
		check_ocl(err);
		exec->exec_queue.finish();
		check_ocl(err);
		cout << "ITERATION " << i << endl;
		cout << "##############################" << endl;
		cl_comm.reset_communicator();
		reset_discovery(exec, d_ctx_mem, false);
		reset_barrier(exec, d_bar);
		reset_persistent_task(exec);
		restart_scheduler(&s_ctx);
		reset_non_persistent();
		reset_persistent_task(exec);
		if (i == 0) {
			cl_comm.set_record_groups_time_data(true);
		}
		else {
			cl_comm.set_record_groups_time_data(false);
		}
		execute_merged_iteration(exec, cl_comm, num_wgs, workgroups_for_non_persistent, error_non_persistent);
		if (!check_persistent_task(exec)) {
			error_persistent = 1;
		}
		ostringstream convert;
		convert << "_" << i << ".txt";
		string it = convert.str();
		cl_comm.print_groups_time_data((FLAGS_output_timestamp_executing_groups + it).c_str());
		cl_comm.print_response_exec_data((FLAGS_output_timestamp_non_persistent + it).c_str());
		cl_comm.print_response_and_execution_times((FLAGS_output_non_persistent_duration + it).c_str());
		cl_comm.print_summary_file((FLAGS_output_summary + it).c_str());
		cl_comm.print_summary();
		times.push_back(cl_comm.get_persistent_time());
		cout << endl;
		CL_Communicator::my_sleep(1000);
	}

	free_scheduler_ctx(exec, &s_ctx);
	clean_persistent_task(exec);
	clean_non_persistent_task(exec);

	cout << endl << "error non persistent: " << error_non_persistent << endl;
	cout << endl << "error persistent: " << error_non_persistent << endl << endl;

	cout << "stats for persistent tasks" << endl;
	cout << "Total time: " << CL_Communicator::reduce_times_ms(times) << " ms" << endl;
	cout << "Mean time: " << CL_Communicator::get_average_time_ms(times) << " ms" << endl;
	cout << "Max time: " << CL_Communicator::max_times_ms(times) << " ms" << endl;
	cout << "Min time: " << CL_Communicator::min_times_ms(times) << " ms" << endl;
	cout << "Standard Deviation: " << CL_Communicator::std_dev_times_ms(times) << " ms" << endl;

	return;
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

	CL_Execution exec;

	exec.exec_device = devices[FLAGS_platform_id][FLAGS_device_id];

	printf("Using GPU: %s\n", exec.getExecDeviceName().c_str());



	cl::Context context(exec.exec_device);
	exec.exec_context = context;
	//cl::CommandQueue queue(exec.exec_context, CL_QUEUE_PROFILING_ENABLE);
	cl::CommandQueue queue(exec.exec_context, CL_QUEUE_PROFILING_ENABLE);
	exec.exec_queue = queue;

	if (FLAGS_run_non_persistent > 0) {
		cout << "Running solo non persistent task" << endl;
		run_non_persistent(&exec);
		return 0;
	}
	else if (FLAGS_run_persistent > 0) {
		cout << "Running solo persistent task" << endl;
		run_persistent(&exec);
		return 0;
	}
	else {
		cout << "Running merged persistent and non_persistent task" << endl;
		run_merged(&exec);
		return 0;
	}



}
