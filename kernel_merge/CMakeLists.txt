project(kernel_merge)
cmake_minimum_required(VERSION 2.8.7)

###############################################################################
# Setup includes
###############################################################################
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_BINARY_DIR})

set(LLVM_SRC "" CACHE PATH "LLVM source directory")
set(LLVM_BUILD "" CACHE PATH "LLVM build directory")
set(LLVM_BUILD_TYPE "" CACHE STRING "LLVM build type")

if (NOT EXISTS "${LLVM_SRC}/include/llvm")
  message(FATAL_ERROR "Invalid LLVM source directory: ${LLVM_SRC}")
endif()

set(LLVM_LIBDIR "${LLVM_BUILD}/${LLVM_BUILD_TYPE}/lib")
if (NOT EXISTS "${LLVM_LIBDIR}")
  message(FATAL_ERROR "Invalid LLVM build directory: ${LLVM_BUILD}")
endif()

set(LLVM_CXXFLAGS "\"/I${LLVM_SRC}/include\" \"/I${LLVM_BUILD}/include\" \"/I${LLVM_SRC}/tools/clang/include\" \"/I${LLVM_BUILD}/tools/clang/include\" \"/I${LLVM_SRC}/tools/clang/lib\" -D_SCL_SECURE_NO_WARNINGS -wd4141 -wd4146 -wd4244 -wd4291 -wd4624 -wd4800")
set(LLVM_LDFLAGS "")
file(GLOB LLVM_LIBS "${LLVM_LIBDIR}/*.lib")

add_executable(kernel_merge
  src/KernelMerge.cpp
  src/ProcessPersistentKernelVisitor.cpp
  src/ProcessNonPersistentKernelVisitor.cpp
  include/ProcessPersistentKernelVisitor.h
  include/ProcessNonPersistentKernelVisitor.h
  include/ProcessKernelVisitor.h
  include/KernelInfo.h
)

set_target_properties(kernel_merge
    PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}"
)

target_link_libraries(kernel_merge PUBLIC ${LLVM_LIBS} ${LLVM_SYS_LIBS} version.lib)
