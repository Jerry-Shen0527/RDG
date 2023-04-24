#pragma once

#ifdef RDG_WITH_CUDA
bool importBufferToBuffer(
    nvrhi::IBuffer* buffer_handle,
    void*& devPtr,
    uint32_t cudaUsageFlags,
    nvrhi::IDevice* device);

CUdeviceptr mapBufferToCUDABuffer(nvrhi::IBuffer* pBuf, uint32_t cudaUsageFlags, nvrhi::IDevice* device);
CUtexObject
mapTextureToCudaTex(nvrhi::ITexture* image_handle, uint32_t cudaUsageFlags, nvrhi::IDevice* device);
#endif
