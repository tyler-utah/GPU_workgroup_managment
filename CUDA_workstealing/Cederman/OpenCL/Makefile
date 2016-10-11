OPENCL_ROOT=$(HOME)/OPENCL

CC = x86_64-w64-mingw32-g++
#CFLAGS = -std=c++11
# For cmake compatibility, root the includes at src/
#INCL = -I/home/admin/GPU_workgroup_managment/src -I$(OPENCL_ROOT)/include
INCL = -I$(OPENCL_ROOT)/include
LIB = -L$(OPENCL_ROOT)/lib

all: octreepart

%.o: %.cc
	$(CC) $(CFLAGS) $(INCL) -c $< -o $@

octreepart: main.cc helper.o lbabp.o octree.o rand.o
	$(CC) $(CFLAGS) $(INCL) $^ -o $@ $(LIB) -lOpenCL

PHONY: clean

clean:
	rm *.o octreepart
