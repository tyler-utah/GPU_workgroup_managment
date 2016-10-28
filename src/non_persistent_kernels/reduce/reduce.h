
DEFINE_string(non_persistent_kernel_file, "pannotia/non_persistent_kernels/device/reduce.cl", "the path the non persistent file");

int graphics_arr_length;
cl_int * h_graphics_buffer;
int arr_min;
cl::Buffer d_graphics_buffer;
cl_int * graphics_result;

const char* non_persistent_app_name() {
	return "reduce";
}

const char* non_persistent_kernel_name() {
	return "MY_reduce";
}

void init_non_persistent_app(CL_Execution *exec) {
	graphics_arr_length = 1048576;
	h_graphics_buffer = (cl_int *)malloc(sizeof(cl_int) * graphics_arr_length);
	arr_min = INT_MAX;
	for (int i = 0; i < graphics_arr_length; i++) {
		int loop_int = rand() + 1;
		if (loop_int < arr_min) {
			arr_min = loop_int;
		}
		h_graphics_buffer[i] = loop_int;
	}

	d_graphics_buffer = cl::Buffer(exec->exec_context, CL_MEM_READ_WRITE, sizeof(cl_int) * graphics_arr_length);
	int err = exec->exec_queue.enqueueWriteBuffer(d_graphics_buffer, CL_TRUE, 0, sizeof(cl_int) * graphics_arr_length, h_graphics_buffer);
	check_ocl(err);
	graphics_result = (cl_int*)clSVMAlloc(exec->exec_context(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 4);
	*graphics_result = INT_MAX;
	free(h_graphics_buffer);
}

int set_non_persistent_app_args(int arg_index, cl::Kernel k) {
	// Set the args for graphics kernel
	int err = k.setArg(arg_index, d_graphics_buffer);
	arg_index++;
        check_ocl(err);
	err = k.setArg(arg_index, graphics_arr_length);
	arg_index++;
        check_ocl(err);
	err = clSetKernelArgSVMPointer(k(), arg_index, graphics_result);
	arg_index++;
	check_ocl(err);
	return arg_index;
}

void reset_non_persistent() {
	*graphics_result = INT_MAX;
}

bool check_non_persistent_task() {
	return arr_min == *graphics_result;
}

void clean_non_persistent_task(CL_Execution *exec) {
	clSVMFree(exec->exec_context(), graphics_result);
}
