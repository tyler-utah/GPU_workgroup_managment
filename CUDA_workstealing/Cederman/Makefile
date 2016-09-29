

SRC=rand.cu helper.cu main.cu

all:
	nvcc $(SRC) -o main

clean:
	rm -rf *.ptx main *.o *~  

#to run: $ ./main 400 20 abp 4096 50
