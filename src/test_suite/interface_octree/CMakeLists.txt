cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../scheduler_rt/rt_host/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../scheduler_rt/rt_common/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../base/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/)

add_executable(octree
  ../../host_skeleton/skel.cpp
  host/octree.h
  host/rand.cc
)
target_link_libraries(octree
  base_commandlineflags
  opencl_opencl
  base_profile
  base_file
)

add_definitions(-DPERSISTENT_OCTREE)
