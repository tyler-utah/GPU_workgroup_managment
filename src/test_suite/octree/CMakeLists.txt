cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
add_executable(test_suite_octree_octree
helper.cc
helper.h
main.cc
octree.cc
octree.h
shared_host_side.h
rand.cc
)

file(COPY device/octree_kernel.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(test_suite_octree_octree
  opencl_opencl
)

