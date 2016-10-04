cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../uvm_tests/test1/include/ocl/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../uvm_tests/test1/include/rt/)
add_executable(first_persistent
  host/main.cpp
)
target_link_libraries(first_persistent
  base_commandlineflags
  opencl_opencl
)

add_definitions(-DKERNEL_FILE=${CMAKE_CURRENT_SOURCE_DIR}/device/mega_kernel.cl)
