
__kernel void copy(__global float * dst, __global float * src) {
	 dst[get_global_id(0)] = src[get_global_id(0)];
}
