cmake_minimum_required(VERSION 3.0)

project(cuda_test)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

cuda_add_executable(hello hello/hello.cu)
cuda_add_executable(vecAdd vecAdd/vecAdd.cu)
cuda_add_executable(query query/query.cu)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)



