#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>
#include <sched.h>

using namespace std;

#define INFY 2000000000

#define TPB 256

int THREADS_PER_BLOCK = TPB;

extern "C" double* cuda_initialize(double **u_res, int nx, double *device_u_res){
    double *host_u_res = new double[3*(nx+2)];
    for(int i=0; i<2; i++){
        for(int j=0; j<nx+2; j++){
            host_u_res[i*(nx+2)+j] = u_res[i][j];
        }
    }    

    cudaMalloc((void **)&device_u_res, 3*(nx+2)*sizeof(double)); 

    // Copy from host to device localPointArray
    cudaMemcpy(device_u_res, host_u_res, \
     2*(nx+2)*sizeof(double), cudaMemcpyHostToDevice);

    delete host_u_res;

    return device_u_res;

}

extern "C" void setDeviceProps(int rank, int size){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    int device = rank % nDevices;
    cudaSetDevice(device);
}

extern "C" void getDeviceProps(int rank, int size){

    int device;

    cudaGetDevice(&device);

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);

    printf("My GPU device ID is: %d out of GPU devices: %d in host: %s for MPI rank : %d out of size: %d\n", device, nDevices, hostname, rank, size);

}

__global__ void cuda_compute_next_step(int nx, double cfl, double *device_u_res){

    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx>=nx){
        return;
    }

    device_u_res[2*(nx+2)+(idx+1)] = 2*device_u_res[1*(nx+2)+(idx+1)] -\
    device_u_res[0*(nx+2)+(idx+1)] + (cfl*cfl)*(device_u_res[1*(nx+2)+(idx)] -\
     2*device_u_res[1*(nx+2)+(idx+1)] + device_u_res[1*(nx+2)+(idx+2)]);
    
}

extern "C" void cuda_update_first_and_last(double **u_res, int nx, int pos, double *device_u_res){
    double first[2] = {u_res[pos][0], u_res[pos][nx+1]};
    cudaMemcpy(&device_u_res[pos*(nx+2)], &first[0], \
     sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_u_res[pos*(nx+2)+nx+1], &first[1], \
     sizeof(double), cudaMemcpyHostToDevice);
}

extern "C" void cuda_get_first_last(double **u_res, int nx, double *device_u_res){
    cudaMemcpy(&u_res[1][1], &device_u_res[1*(nx+2)+1] , \
        sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(&u_res[1][nx], &device_u_res[1*(nx+2)+nx] , \
        sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void cuda_kernel_shift(int nx, double *device_u_res){

    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx>=nx+2){
        return;
    }

    device_u_res[0*(nx+2)+(idx)] = device_u_res[1*(nx+2)+(idx)];
    
    device_u_res[1*(nx+2)+(idx)] = device_u_res[2*(nx+2)+(idx)];
}

extern "C" void cuda_back_to_host(double **u_res, int last_pos, int nx, double *device_u_res){

    double *host_u_res = new double[3*(nx+2)]; 
    
    cudaMemcpy(host_u_res, device_u_res, \
     3*(nx+2)*sizeof(double), cudaMemcpyDeviceToHost);

    for(int i=0; i<3; i++){
        for(int j=0; j<nx+2; j++){
            u_res[i][j] = host_u_res[i*(nx+2)+j];
        }
    }  
}

extern "C" void cuda_shift_values(double **u_res, int nx, double *device_u_res){
    int num_blocks = (nx+2 + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    
    cuda_kernel_shift<<<num_blocks, THREADS_PER_BLOCK>>>(nx, device_u_res);
}



extern "C" void cuda_compute_values_next_time_step(double **u_res, int nx, double cfl, double *device_u_res){
    int num_blocks = (nx + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    
    cuda_compute_next_step<<<num_blocks, THREADS_PER_BLOCK>>>(nx, cfl, device_u_res);
}


extern "C" void cudaDeInitialize(double *device_u_res){

    cudaFree(device_u_res);

}
