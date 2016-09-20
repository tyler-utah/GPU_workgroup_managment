cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ocl/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/rt/)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/cmake)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

find_package(OpenCL REQUIRED)

include_directories(${OPENCL_INCLUDE_DIR})

set (UVM_TEST1 "uvm_test1")

add_executable(${UVM_TEST1}
  src/host/main.cpp
)

add_definitions(-DKERNEL_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/device/megaKernel.cl)

target_link_libraries(${UVM_TEST1} ${OPENCL_LIBRARIES})

