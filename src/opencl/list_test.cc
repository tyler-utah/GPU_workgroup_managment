#include "base/commandlineflags.h"
#include <iostream>
#include "opencl/interface.h"

OPENCL_LIST;

int main(int argc, char **argv) {
  //std::cout << FLAGS_opencl_list << std::endl;
  flags::ParseCommandLineFlags(&argc, &argv, true);
}
