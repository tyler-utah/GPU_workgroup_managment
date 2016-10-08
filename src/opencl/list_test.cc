#include "base/commandlineflags.h"
//#include "opencl/interface.h"

int main(int argc, char **argv) {
  flags::ParseCommandLineFlags(&argc, &argv, true);
}
