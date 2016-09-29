
#include "discovery.cl"

__kernel void mega_kernel(__global Discovery_ctx *d_ctx){

	DISCOVERY_PROTOCOL(d_ctx);

}
//