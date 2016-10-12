cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ocl/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../scheduler_rt/rt_host/)
add_executable(uvm_test1
  src/host/main.cpp
)
target_link_libraries(uvm_test1
  base_commandlineflags
  opencl_opencl
)

add_definitions(-DKERNEL_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/device/megaKernel.cl)
