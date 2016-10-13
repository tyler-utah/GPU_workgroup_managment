cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../scheduler_rt/rt_host/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../base/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/)

add_executable(first_persistent
  host/main.cpp
)
target_link_libraries(first_persistent
  base_commandlineflags
  opencl_opencl
  base_profile
)

add_definitions(-DKERNEL_FILE=${CMAKE_CURRENT_SOURCE_DIR}/device/mega_kernel.cl)
