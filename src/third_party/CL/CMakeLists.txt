cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(cl
  cl.h
  cl_ext.h
  cl_platform.h
  cl2.hpp
  opencl.h
)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCL REQUIRED)
target_link_libraries(cl ${OPENCL_LIBRARIES})
set_target_properties(cl PROPERTIES LINKER_LANGUAGE CXX)
