cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# ===== base_commandlineflags
add_library(base_commandlineflags INTERFACE)
# Header: commandlineflags.h
target_link_libraries(base_commandlineflags INTERFACE
  gflags_static
)

# ===== base_file
add_library(base_file
  file.cc
  file.h
)
target_link_libraries(base_file
  base_platform
)

# ===== base_platform
add_library(base_platform
  platform.cc
  platform.h
)
target_compile_definitions(base_platform PRIVATE
  PROJECT_ROOT=${GPU_workgroup_management_SOURCE_DIR}/
)



add_executable(file_test
test_file_main.cc)
target_link_libraries(file_test
  base_file
)
