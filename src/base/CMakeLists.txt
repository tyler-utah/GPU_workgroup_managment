cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

add_library(base_commandlineflags
  INTERFACE
)

target_include_directories(base_commandlineflags INTERFACE commandlineflags.h)

target_link_libraries(base_commandlineflags INTERFACE
  gflags_static
)

#set_target_properties(base_commandlineflags PROPERTIES LINKER_LANGUAGE CXX)
