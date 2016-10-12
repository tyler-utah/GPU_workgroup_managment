cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
add_executable(test_suite_checkRequirements_checkRequirements
  main.cc
)

target_link_libraries(test_suite_checkRequirements_checkRequirements
  opencl_opencl
)
