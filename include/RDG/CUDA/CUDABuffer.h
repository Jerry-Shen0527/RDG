#pragma once

#include "cuda_runtime_api.h"
#include "CUDAException.h"

#define MallocCUDABuffer(size, name)                                        \
    if ((name) != 0)                                                        \
        cudaFree((name));                                                   \
    CUDA_CHECK(cudaMalloc((void**)&(name), size* PointedTypeSize((name)))); \
    CUDA_CHECK(cudaMemset((name), 0, size* PointedTypeSize((name))));

#define MallocCUDABufferManaged(size, name)                                        \
    if ((name) != 0)                                                               \
        cudaFree((name));                                                          \
    CUDA_CHECK(cudaMallocManaged((void**)&(name), size* PointedTypeSize((name)))); \
    CUDA_CHECK(cudaMemset((name), 0, size* PointedTypeSize((name))));

template<typename T>
constexpr size_t PointedTypeSize(const T* ptr)
{
    return sizeof(T);
}