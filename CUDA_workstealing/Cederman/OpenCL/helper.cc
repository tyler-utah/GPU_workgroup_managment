#include <CL/cl.hpp>
#include <iostream>

/*---------------------------------------------------------------------------*/

void checkErr(cl_int err, const char * name) {
  if (err != CL_SUCCESS) {
    std::cerr << "ERROR: " << name  << " (" << err << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

/*---------------------------------------------------------------------------*/
