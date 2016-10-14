clang -E mega_kernel.cl -I ..\common -I ..\..\..\scheduler_rt\rt_device -DCL_INT_TYPE=int -DATOMIC_CL_INT_TYPE=atomic_int 
