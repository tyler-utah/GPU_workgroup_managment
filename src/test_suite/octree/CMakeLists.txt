cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
add_executable(test_suite_octree_octree
helper.cc
helper.h
lbabp.cc
lbabp.h
main.cc
octree.cc
octree.h
octree_host_dev_shared.h
rand.cc
)

target_link_libraries(test_suite_octree_octree
  opencl_opencl
)

