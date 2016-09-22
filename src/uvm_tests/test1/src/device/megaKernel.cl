 
#define GPU_WAIT 0
#define GPU_ADD 1
#define GPU_MULT 2
#define GPU_QUIT 3
 
 __kernel void mega_kernel(__global atomic_int *flag, __global int* data1, __global int* data2, __global int* result) {
 
	while(true) {
		int local_flag = atomic_load_explicit(flag, memory_order_acquire, memory_scope_all_svm_devices);
		if (local_flag == GPU_QUIT) {
			break;
		}
		if (local_flag == GPU_ADD) {
			int local_data1 = *data1;
			int local_data2 = *data2;
			*result = local_data1 + local_data2;
			atomic_store_explicit(flag, GPU_WAIT, memory_order_release, memory_scope_all_svm_devices);
		}
		if (local_flag == GPU_MULT) {
			int local_data1 = *data1;
			int local_data2 = *data2;
			*result = local_data1 * local_data2;
			atomic_store_explicit(flag, GPU_WAIT, memory_order_release, memory_scope_all_svm_devices);
		}
	}
	
 }
 //