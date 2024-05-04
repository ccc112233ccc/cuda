#pragma once
#include<stdio.h>


cudaError_t cudaErrorCheck(cudaError_t error, const char* file, const int line) {
    if(error != cudaSuccess) {
        printf("CUDA Error: [%s] in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(-1);
    }
    return error;
}

#define CUDA_ERROR_CHECK(error) (cudaErrorCheck(error, __FILE__, __LINE__))