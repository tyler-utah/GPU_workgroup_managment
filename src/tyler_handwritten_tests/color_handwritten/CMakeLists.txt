cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../scheduler_rt/rt_host/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../base/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/)

add_library(graph_parse host/parse.h host/parse.cpp)

add_executable(color_handwritten
  host/main.cpp
  common/restoration_ctx.h
)
target_link_libraries(color_handwritten
  base_commandlineflags
  opencl_opencl
  base_profile
  base_file
  graph_parse
)

add_definitions(-DKERNEL_FILE=${CMAKE_CURRENT_SOURCE_DIR}/device/mega_kernel.cl)
