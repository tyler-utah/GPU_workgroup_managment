
CC = g++
CFLAGS = -Wall -Werror

#OPENCL=$(HOME)/OPENCL
OPENCL = /vol/cuda/7.5.18
# For cmake compatibility, root the includes at src/
#INCL = -I/home/admin/GPU_workgroup_managment/src -I$(OPENCL)/include
INCL = -I$(OPENCL)/include
#LIB = -L$(OPENCL)/lib
LIB = -L$(OPENCL)/lib64

all: octreepart run

%.o: %.cc
	$(CC) $(CFLAGS) $(INCL) -c $< -o $@

octreepart: main.cc helper.o lbabp.o octree.o rand.o
	$(CC) $(CFLAGS) $(INCL) $^ -o $@ $(LIB) -lOpenCL

PHONY: clean run

clean:
	rm -f *.o octreepart

run:
	env LD_PRELOAD=$(OPENCL)/lib64/libOpenCL.so ./octreepart 128 20 abp 1024 50
