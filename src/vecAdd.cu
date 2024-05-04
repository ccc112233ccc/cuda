#include<stdio.h>
#include"common.cuh"


__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 1D grid, 1D block
    while(idx < n) {
        c[idx] = a[idx] + b[idx];
        idx += blockDim.x * gridDim.x;
    }
}

void vecAddHost(float *a, float *b, float *c, int n) {
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int n = 10000000;
    int size = n * sizeof(float);


    // Allocate memory for host
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    // Allocate memory for device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Initialize host memory
    for(int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Copy host memory to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    vecAdd<<<1024, 1024>>>(d_a, d_b, d_c, n);
    CUDA_ERROR_CHECK(cudaPeekAtLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // print time for seconds
    printf("GPU Time: %f\n", milliseconds / 1000);

    // Copy device memory to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Check result
    for(int i = 0; i < n; i++) {
        if(c[i] != a[i] + b[i]) {
            printf("Error at index %d\n", i);
            break;
        }
    }

    // Test host function

    // Start timer
    int start_time = clock();
    vecAddHost(a, b, c, n);
    int end_time = clock();

    printf("CPU Time: %f\n", (end_time - start_time) / (float)CLOCKS_PER_SEC);

    // Free memory
    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

