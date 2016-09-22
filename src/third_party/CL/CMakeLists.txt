cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

add_library(cl
  cl.h
  cl_d3d10.h
  cl_d3d11.h
  cl_dx9_media_sharing.h
  cl_egl.h
  cl_ext.h
  cl_gl.h
  cl_gl_ext.h
  cl_platform.h
  cl2.hpp
  opencl.h
)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCL REQUIRED)
target_link_libraries(cl ${OPENCL_LIBRARIES})
target_include_directories(cl PUBLIC ${GPU_workgroup_management_SOURCE_DIR}/third_party)
set_target_properties(cl PROPERTIES LINKER_LANGUAGE CXX)
