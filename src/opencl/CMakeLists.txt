cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

add_library(opencl_opencl
  INTERFACE
)

target_include_directories(opencl_opencl INTERFACE opencl.h)

target_link_libraries(opencl_opencl INTERFACE
  cl
)
#set_target_properties(opencl_opencl PROPERTIES LINKER_LANGUAGE CXX)
