#pragma once
#include"RDG/CUDA/CUDARandom.cuh"

#include "RDG/CUDA/CUDABuffer.h"
#include "RDG/CUDA/GPUParallel.cuh"

unsigned* generate_seeds(unsigned size)
{
    unsigned* ret = 0;
    MallocCUDABuffer(size, ret);
    GPUParallelFor(
        "Setting seeds", size, GPU_LAMBDA_Ex(int i) { unsigned seed = i;
            rnd(seed);
            rnd(seed);
            rnd(seed);
            rnd(seed);
            rnd(seed);
            ret[i] = seed;
        });
    return ret;
}
