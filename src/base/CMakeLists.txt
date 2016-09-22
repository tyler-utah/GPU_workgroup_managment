cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

add_library(base_commandlineflags
  commandlineflags.h
)
target_link_libraries(base_commandlineflags
  gflags
)
set_target_properties(base_commandlineflags PROPERTIES LINKER_LANGUAGE CXX)
