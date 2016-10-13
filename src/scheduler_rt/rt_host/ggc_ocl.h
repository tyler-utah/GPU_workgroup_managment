#pragma once

#include "stdlib.h"
#include <string>

#define MAX_DISCOVERY_LAUNCH 1024

#define XSTR(a) STR(a)
#define STR(a) #a

void check_ocl_error(const int e, const char *file, const int line) {
  if (e < 0) {
    printf("%s:%d: error (%d)\n", file, line, e);
    exit(1);
  }
}

#define check_ocl(err) check_ocl_error(err, __FILE__, __LINE__)
