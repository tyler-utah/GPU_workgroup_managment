@echo off

RelWithDebInfo\kernel_merge.exe %1 %2 -- -I C:\Users\afd\llvm39-build\Release\lib\clang\3.9.1\include -I ..\kernel_merge\stubs\premerge -include opencl-c.h -include global_barrier.h -cl-std=CL2.0

clang -cc1 -finclude-default-header -I C:\Users\afd\llvm39-build\Release\lib\clang\3.9.1\include -cl-std=CL2.0 -I . -I ..\src\scheduler_rt\rt_device  -include ..\src\scheduler_rt\rt_common\cl_types.h merged.cl

clang-format -style="{SortIncludes: false}" -i merged.cl
