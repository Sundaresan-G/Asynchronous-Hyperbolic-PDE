CC=/apps/cuda/bin/nvcc

all: main.out

main.out: device.o host.o
	mpic++ -o main.out host.o device.o -lgomp -L/apps/cuda/lib64 -lcudart
	
device.o: device.cu
	$(CC) -c -Xcompiler -fopenmp -o device.o device.cu

host.o: cuda_async_mpi.cpp
	mpic++ -c -o host.o cuda_async_mpi.cpp 

clean:
	rm -f main.out host.o device.o
