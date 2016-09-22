cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(CL)
SET(GFLAGS_IS_SUBPROJECT TRUE)
SET(GFLAGS_INCLUDE_DIR "third_party/gflags")
SET(LIBRARY_INSTALL_DIR "third_party/gflags")
add_subdirectory(gflags)
