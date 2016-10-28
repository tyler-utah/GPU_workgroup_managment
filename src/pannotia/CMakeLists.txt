cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../scheduler_rt/rt_host/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../scheduler_rt/rt_common/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../base/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/graph_apps/color/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/graph_parser/)


add_library(pannotia_graph_parse graph_parser/parse.h graph_parser/parse.cpp)

add_executable(pannotia_color
  ../host_skeleton/skel.cpp
  graph_apps/color/color.h
)
target_link_libraries(pannotia_color
  base_commandlineflags
  opencl_opencl
  base_profile
  base_file
  pannotia_graph_parse
)

add_definitions(-DPERSISTENT_PANNOTIA_COLOR)
