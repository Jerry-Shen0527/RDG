#pragma once
#include <vector_types.h>
#include "nvrhi/nvrhi.h"

void BlitLinearBufferToSurface(
    float4* buffer,
    nvrhi::cudaSurfaceObject_t surfaceObject,
    int width,
    int height);

void BlitLinearBufferToSurface(
    float3* buffer,
    nvrhi::cudaSurfaceObject_t surfaceObject,
    int width,
    int height);


void ComposeChannels(float4* target, float* x, float* y, float* z, float* w, int size);
void ComposeChannels(float4* target, float* x, float* y, float* z, int size);
void ComposeChannels(float3* target, float* x, float* y, float* z, int size);