cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# ===== opencl_opencl
add_library(opencl_opencl INTERFACE)
# Header: opencl.h
target_link_libraries(opencl_opencl INTERFACE
  cl
)
