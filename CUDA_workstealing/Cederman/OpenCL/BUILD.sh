#!/bin/sh

CC="g++"
#FLAGS="-g"
LIB="-lOpenCL"
OPENCL="/vol/cuda/7.5.18"
#OPENCL="/vol/cuda/8.0.44"
INCL="-I$OPENCL/include"
LIBPATH="-L$OPENCL/lib64"

set -e
set -x

$CC $FLAGS $INCL -c rand.cc -o rand.o
$CC $FLAGS $INCL -c helper.cc -o helper.o
$CC $FLAGS $INCL -c octree.cc -o octree.o
$CC $FLAGS $INCL -c lbabp.cc -o lbabp.o
$CC $FLAGS $INCL main.cc rand.o helper.o octree.o lbabp.o -o octreepart $LIBPATH $LIB

set +x
echo "===== BUILD SUCCESS ====="

COMMAND="env LD_PRELOAD=$OPENCL/lib64/libOpenCL.so ./octreepart 128 20 abp 1024 50"

echo "===== RUN COMMAND ====="
echo "$COMMAND"
$COMMAND
echo "===== RUN IS OVER ====="
