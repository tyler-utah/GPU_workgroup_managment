cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../scheduler_rt/rt_host/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../scheduler_rt/rt_common/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../base/)

add_executable(lonestar_bfs
  ../host_skeleton/skel.cpp
  graph_apps/bfs/bfs.h
)

target_link_libraries(lonestar_bfs
  base_commandlineflags
  opencl_opencl
  base_profile
  base_file
)

target_compile_options(lonestar_bfs PUBLIC -DPERSISTENT_LONESTAR_BFS -DNON_PERSISTENT_MATMULT)
target_include_directories(lonestar_bfs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/graph_apps/bfs/)

add_executable(lonestar_sssp
  ../host_skeleton/skel.cpp
  graph_apps/sssp/sssp.h
)

target_link_libraries(lonestar_sssp
  base_commandlineflags
  opencl_opencl
  base_profile
  base_file
)

target_compile_options(lonestar_sssp PUBLIC -DPERSISTENT_LONESTAR_SSSP -DNON_PERSISTENT_MATMULT)
target_include_directories(lonestar_sssp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/graph_apps/sssp/)
