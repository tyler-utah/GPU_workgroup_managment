cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../scheduler_rt/rt_host/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../scheduler_rt/rt_common/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../base/)
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

target_compile_options(pannotia_color PUBLIC -DPERSISTENT_PANNOTIA_COLOR)
target_include_directories(pannotia_color PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/graph_apps/color/)

add_executable(pannotia_mis
  ../host_skeleton/skel.cpp
  graph_apps/mis/mis.h
)

target_link_libraries(pannotia_mis
  base_commandlineflags
  opencl_opencl
  base_profile
  base_file
  pannotia_graph_parse
)

target_compile_options(pannotia_mis PUBLIC -DPERSISTENT_PANNOTIA_MIS)
target_include_directories(pannotia_mis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/graph_apps/mis/)


add_executable(pannotia_sssp
  ../host_skeleton/skel.cpp
  graph_apps/sssp/sssp.h
)

target_link_libraries(pannotia_sssp
  base_commandlineflags
  opencl_opencl
  base_profile
  base_file
  pannotia_graph_parse
)

target_compile_options(pannotia_sssp PUBLIC -DPERSISTENT_PANNOTIA_SSSP)
target_include_directories(pannotia_sssp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/graph_apps/sssp/)



