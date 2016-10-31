cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../scheduler_rt/rt_host/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../scheduler_rt/rt_common/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../base/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

add_executable(os_freq
  host/os_freq.cpp
)
target_link_libraries(os_freq
  base_commandlineflags
  opencl_opencl
  base_profile
  base_file
)
