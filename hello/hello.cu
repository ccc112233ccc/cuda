#include<stdio.h>
#include<iostream>

__global__ void helloFromGPU(void) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int threadId = (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int id = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId;
    printf("Hello from GPU! BlockId: %d, ThreadId: %d, Id: %d\n", blockId, threadId, id);

}


int hello() {
    // Launch the kernel
    dim3 block(2, 2);
    dim3 grid(2, 2);
    helloFromGPU<<<grid, block>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    return 0;
}
