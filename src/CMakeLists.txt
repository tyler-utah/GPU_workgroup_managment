cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(GPU_workgroup_management)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
  BEFORE
  ${GPU_workgroup_management_SOURCE_DIR}
  ${GPU_workgroup_management_BINARY_DIR}
)

link_directories(
  ${GPU_workgroup_management_BINARY_DIR}
)

add_subdirectory(base)
add_subdirectory(opencl)
add_subdirectory(third_party)
add_subdirectory(uvm_tests)
add_subdirectory(tyler_handwritten_tests)
add_subdirectory(test_suite)
add_subdirectory(pannotia)
