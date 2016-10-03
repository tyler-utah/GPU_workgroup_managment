cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# ===== base_commandlineflags
add_library(base_commandlineflags INTERFACE)
# Header: commandlineflags.h
target_link_libraries(base_commandlineflags INTERFACE
  gflags_static
)
