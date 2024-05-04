#include<stdio.h>
#include"common.cuh"

const int N = 1024 * 1000; // 1M elements
const int ThreadPerBlock = 1024;

double sum_squares(int x) {
    return 1.0 * x * (x + 1.0) * (2.0 * x + 1.0) / 6.0;
}


__global__ void vecProduct(double *a, double *b, double *c, int n) {
    __shared__ double cache[ThreadPerBlock]; // Shared memory for each block
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // Global thread ID
    int cacheIndex = threadIdx.x; // Cache index

    double temp = 0; // Temporary variable to store the product of a[tid] and b[tid]
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp; // Store the product in the cache

    __syncthreads(); // Synchronize threads in the block

    // Perform reduction in the cache
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}



const int blockPerGrid = (N + ThreadPerBlock - 1) / ThreadPerBlock; // Number of blocks in the grid, rounded up


int main() {
    double *a, *b, *c, result;
    double *d_a, *d_b, *d_c;

    // Allocate memory on the host
    a = (double *)malloc(N * sizeof(double));
    b = (double *)malloc(N * sizeof(double));
    c = (double *)malloc(blockPerGrid * sizeof(double));

    // Allocate memory on the device
    cudaMalloc(&d_a, N * sizeof(double));
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_c, blockPerGrid * sizeof(double));

    // Initialize vectors a and b
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Copy data from host to device
    cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    vecProduct<<<blockPerGrid, ThreadPerBlock>>>(d_a, d_b, d_c, N);

    // Copy data from device to host
    cudaMemcpy(c, d_c, blockPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate the result
    result = 0;
    for (int i = 0; i < blockPerGrid; i++) {
        result += c[i];
    }


    printf("Expected: %f, Got: %f\n", sum_squares(N - 1), result);

    // Free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on the host
    free(a);
    free(b);
    free(c);

    return 0;
}

