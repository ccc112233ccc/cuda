#include<stdio.h>
#include"../common.cuh"


int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %lu\n", prop.totalGlobalMem);
    printf("Shared memory per block: %lu\n", prop.sharedMemPerBlock);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Total constant memory: %lu\n", prop.totalConstMem);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Kernel execution timeout enabled: %s\n", prop.kernelExecTimeoutEnabled ? "true" : "false");

    return 0;
}