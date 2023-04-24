#include "RDG/OptiX/OptiXScene.h"

#include <iostream>

#include "RDG/CUDA/CUDABuffer.h"
#include "RDG/CUDA/CUDAExternal.h"
#include "donut_math_types.h"
#include "donut/shaders/bindless.h"
#include "donut/shaders/skinning_cb.h"

namespace donut::engine
{
    struct Scene::Resources
    {
        std::vector<MaterialConstants> materialData;
        std::vector<GeometryData> geometryData;
        std::vector<InstanceData> instanceData;
    };

    inline void
    AppendBufferRange(nvrhi::BufferRange& range, size_t size, uint64_t& currentBufferSize)
    {
        range.byteOffset = currentBufferSize;
        range.byteSize = size;
        currentBufferSize += range.byteSize;
    }

    void OptiXScene::createTextureDescriptor(
        std::shared_ptr<LoadedTexture> texture)
    {
        if (texture)
        {
            CUtexObject tex = mapTextureToCudaTex(texture->texture, 0, m_Device);
            createDescriptor(texture->bindlessDescriptor, tex, BindlessTextures);
        }
    }

    void OptiXScene::CreateMeshBuffers(nvrhi::ICommandList* commandList)
    {
        using namespace math;
        for (const auto& mesh : m_SceneGraph->GetMeshes())
        {
            auto buffers = mesh->buffers;

            if (!buffers)
                continue;

            if (!buffers->indexData.empty() && !buffers->indexBuffer)
            {
                nvrhi::BufferDesc bufferDesc;
                bufferDesc.isIndexBuffer = true;
                bufferDesc.byteSize = buffers->indexData.size() * sizeof(uint32_t);
                bufferDesc.debugName = "IndexBuffer";
                bufferDesc.canHaveTypedViews = true;
                bufferDesc.canHaveRawViews = true;
                bufferDesc.format = nvrhi::Format::R32_UINT;
                bufferDesc.isAccelStructBuildInput = m_RayTracingSupported;
                bufferDesc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;

                buffers->indexBuffer = m_Device->createBuffer(bufferDesc);

                if (m_DescriptorTable)
                {
                    buffers->indexBufferDescriptor = std::make_shared<DescriptorHandle>(
                        m_DescriptorTable->CreateDescriptorHandle(
                            nvrhi::BindingSetItem::RawBuffer_SRV(0, buffers->indexBuffer)));
                    auto cudaPtr = mapBufferToCUDABuffer(buffers->indexBuffer, 0, m_Device);
                    createDescriptor(*buffers->indexBufferDescriptor.get(), cudaPtr, BindlessBuffers);
                }

                commandList->beginTrackingBufferState(
                    buffers->indexBuffer,
                    nvrhi::ResourceStates::Common);

                commandList->writeBuffer(
                    buffers->indexBuffer,
                    buffers->indexData.data(),
                    buffers->indexData.size() * sizeof(uint32_t));
                std::vector<uint32_t>().swap(buffers->indexData);

                nvrhi::ResourceStates state =
                    nvrhi::ResourceStates::IndexBuffer | nvrhi::ResourceStates::ShaderResource;

                if (bufferDesc.isAccelStructBuildInput)
                    state = state | nvrhi::ResourceStates::AccelStructBuildInput;

                commandList->setPermanentBufferState(buffers->indexBuffer, state);
                commandList->commitBarriers();
            }

            if (!buffers->vertexBuffer)
            {
                nvrhi::BufferDesc bufferDesc;
                bufferDesc.isVertexBuffer = true;
                bufferDesc.byteSize = 0;
                bufferDesc.debugName = "VertexBuffer";
                bufferDesc.canHaveTypedViews = true;
                bufferDesc.canHaveRawViews = true;
                bufferDesc.isAccelStructBuildInput = m_RayTracingSupported;

                bufferDesc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;

                if (!buffers->positionData.empty())
                {
                    AppendBufferRange(
                        buffers->getVertexBufferRange(VertexAttribute::Position),
                        buffers->positionData.size() * sizeof(buffers->positionData[0]),
                        bufferDesc.byteSize);
                }

                if (!buffers->normalData.empty())
                {
                    AppendBufferRange(
                        buffers->getVertexBufferRange(VertexAttribute::Normal),
                        buffers->normalData.size() * sizeof(buffers->normalData[0]),
                        bufferDesc.byteSize);
                }

                if (!buffers->tangentData.empty())
                {
                    AppendBufferRange(
                        buffers->getVertexBufferRange(VertexAttribute::Tangent),
                        buffers->tangentData.size() * sizeof(buffers->tangentData[0]),
                        bufferDesc.byteSize);
                }

                if (!buffers->texcoord1Data.empty())
                {
                    AppendBufferRange(
                        buffers->getVertexBufferRange(VertexAttribute::TexCoord1),
                        buffers->texcoord1Data.size() * sizeof(buffers->texcoord1Data[0]),
                        bufferDesc.byteSize);
                }

                if (!buffers->texcoord2Data.empty())
                {
                    AppendBufferRange(
                        buffers->getVertexBufferRange(VertexAttribute::TexCoord2),
                        buffers->texcoord2Data.size() * sizeof(buffers->texcoord2Data[0]),
                        bufferDesc.byteSize);
                }

                if (!buffers->weightData.empty())
                {
                    AppendBufferRange(
                        buffers->getVertexBufferRange(VertexAttribute::JointWeights),
                        buffers->weightData.size() * sizeof(buffers->weightData[0]),
                        bufferDesc.byteSize);
                }

                if (!buffers->jointData.empty())
                {
                    AppendBufferRange(
                        buffers->getVertexBufferRange(VertexAttribute::JointIndices),
                        buffers->jointData.size() * sizeof(buffers->jointData[0]),
                        bufferDesc.byteSize);
                }

                buffers->vertexBuffer = m_Device->createBuffer(bufferDesc);
                if (m_DescriptorTable)
                {
                    buffers->vertexBufferDescriptor = std::make_shared<DescriptorHandle>(
                        m_DescriptorTable->CreateDescriptorHandle(
                            nvrhi::BindingSetItem::RawBuffer_SRV(0, buffers->vertexBuffer)));
                    auto cudaPtr = mapBufferToCUDABuffer(buffers->vertexBuffer, 0, m_Device);
                    createDescriptor(
                        *buffers->vertexBufferDescriptor.get(),
                        cudaPtr,
                        BindlessBuffers);
                }

                commandList->beginTrackingBufferState(
                    buffers->vertexBuffer,
                    nvrhi::ResourceStates::Common);

                if (!buffers->positionData.empty())
                {
                    const auto& range = buffers->getVertexBufferRange(VertexAttribute::Position);
                    commandList->writeBuffer(
                        buffers->vertexBuffer,
                        buffers->positionData.data(),
                        range.byteSize,
                        range.byteOffset);
                    std::vector<float3>().swap(buffers->positionData);
                }

                if (!buffers->normalData.empty())
                {
                    const auto& range = buffers->getVertexBufferRange(VertexAttribute::Normal);
                    commandList->writeBuffer(
                        buffers->vertexBuffer,
                        buffers->normalData.data(),
                        range.byteSize,
                        range.byteOffset);
                    std::vector<uint32_t>().swap(buffers->normalData);
                }

                if (!buffers->tangentData.empty())
                {
                    const auto& range = buffers->getVertexBufferRange(VertexAttribute::Tangent);
                    commandList->writeBuffer(
                        buffers->vertexBuffer,
                        buffers->tangentData.data(),
                        range.byteSize,
                        range.byteOffset);
                    std::vector<uint32_t>().swap(buffers->tangentData);
                }

                if (!buffers->texcoord1Data.empty())
                {
                    const auto& range = buffers->getVertexBufferRange(VertexAttribute::TexCoord1);
                    commandList->writeBuffer(
                        buffers->vertexBuffer,
                        buffers->texcoord1Data.data(),
                        range.byteSize,
                        range.byteOffset);
                    std::vector<float2>().swap(buffers->texcoord1Data);
                }

                if (!buffers->texcoord2Data.empty())
                {
                    const auto& range = buffers->getVertexBufferRange(VertexAttribute::TexCoord2);
                    commandList->writeBuffer(
                        buffers->vertexBuffer,
                        buffers->texcoord2Data.data(),
                        range.byteSize,
                        range.byteOffset);
                    std::vector<float2>().swap(buffers->texcoord2Data);
                }

                if (!buffers->weightData.empty())
                {
                    const auto& range =
                        buffers->getVertexBufferRange(VertexAttribute::JointWeights);
                    commandList->writeBuffer(
                        buffers->vertexBuffer,
                        buffers->weightData.data(),
                        range.byteSize,
                        range.byteOffset);
                    std::vector<float4>().swap(buffers->weightData);
                }

                if (!buffers->jointData.empty())
                {
                    const auto& range =
                        buffers->getVertexBufferRange(VertexAttribute::JointIndices);
                    commandList->writeBuffer(
                        buffers->vertexBuffer,
                        buffers->jointData.data(),
                        range.byteSize,
                        range.byteOffset);
                    std::vector<vector<uint16_t, 4>>().swap(buffers->jointData);
                }

                nvrhi::ResourceStates state =
                    nvrhi::ResourceStates::VertexBuffer | nvrhi::ResourceStates::ShaderResource;

                if (bufferDesc.isAccelStructBuildInput)
                    state = state | nvrhi::ResourceStates::AccelStructBuildInput;

                commandList->setPermanentBufferState(buffers->vertexBuffer, state);
                commandList->commitBarriers();
            }
        }

        for (const auto& skinnedInstance : m_SceneGraph->GetSkinnedMeshInstances())
        {
            const auto& skinnedMesh = skinnedInstance->GetMesh();

            if (!skinnedMesh->buffers)
            {
                skinnedMesh->buffers = std::make_shared<BufferGroup>();

                uint32_t totalVertices = skinnedMesh->totalVertices;

                skinnedMesh->buffers->indexBuffer =
                    skinnedInstance->GetPrototypeMesh()->buffers->indexBuffer;
                skinnedMesh->buffers->indexBufferDescriptor =
                    skinnedInstance->GetPrototypeMesh()->buffers->indexBufferDescriptor;

                const auto& prototypeBuffers = skinnedInstance->GetPrototypeMesh()->buffers;
                const auto& skinnedBuffers = skinnedMesh->buffers;

                size_t skinnedVertexBufferSize = 0;
                assert(prototypeBuffers->hasAttribute(VertexAttribute::Position));

                AppendBufferRange(
                    skinnedBuffers->getVertexBufferRange(VertexAttribute::Position),
                    totalVertices * sizeof(float3),
                    skinnedVertexBufferSize);

                AppendBufferRange(
                    skinnedBuffers->getVertexBufferRange(VertexAttribute::PrevPosition),
                    totalVertices * sizeof(float3),
                    skinnedVertexBufferSize);

                if (prototypeBuffers->hasAttribute(VertexAttribute::Normal))
                {
                    AppendBufferRange(
                        skinnedBuffers->getVertexBufferRange(VertexAttribute::Normal),
                        totalVertices * sizeof(uint32_t),
                        skinnedVertexBufferSize);
                }

                if (prototypeBuffers->hasAttribute(VertexAttribute::Tangent))
                {
                    AppendBufferRange(
                        skinnedBuffers->getVertexBufferRange(VertexAttribute::Tangent),
                        totalVertices * sizeof(uint32_t),
                        skinnedVertexBufferSize);
                }

                if (prototypeBuffers->hasAttribute(VertexAttribute::TexCoord1))
                {
                    AppendBufferRange(
                        skinnedBuffers->getVertexBufferRange(VertexAttribute::TexCoord1),
                        totalVertices * sizeof(float2),
                        skinnedVertexBufferSize);
                }

                if (prototypeBuffers->hasAttribute(VertexAttribute::TexCoord2))
                {
                    AppendBufferRange(
                        skinnedBuffers->getVertexBufferRange(VertexAttribute::TexCoord2),
                        totalVertices * sizeof(float2),
                        skinnedVertexBufferSize);
                }

                nvrhi::BufferDesc bufferDesc;
                bufferDesc.isVertexBuffer = true;
                bufferDesc.byteSize = skinnedVertexBufferSize;
                bufferDesc.debugName = "SkinnedVertexBuffer";
                bufferDesc.canHaveTypedViews = true;
                bufferDesc.canHaveRawViews = true;
                bufferDesc.canHaveUAVs = true;
                bufferDesc.isAccelStructBuildInput = m_RayTracingSupported;
                bufferDesc.keepInitialState = true;
                bufferDesc.initialState = nvrhi::ResourceStates::VertexBuffer;

                skinnedBuffers->vertexBuffer = m_Device->createBuffer(bufferDesc);

                if (m_DescriptorTable)
                {
                    skinnedBuffers->vertexBufferDescriptor = std::make_shared<DescriptorHandle>(
                        m_DescriptorTable->CreateDescriptorHandle(
                            nvrhi::BindingSetItem::RawBuffer_SRV(0, skinnedBuffers->vertexBuffer)));
                }
            }

            if (!skinnedInstance->jointBuffer)
            {
                nvrhi::BufferDesc jointBufferDesc;
                jointBufferDesc.debugName = "JointBuffer";
                jointBufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
                jointBufferDesc.keepInitialState = true;
                jointBufferDesc.canHaveRawViews = true;
                jointBufferDesc.byteSize = sizeof(dm::float4x4) * skinnedInstance->joints.size();
                skinnedInstance->jointBuffer = m_Device->createBuffer(jointBufferDesc);
            }

            if (!skinnedInstance->skinningBindingSet)
            {
                const auto& prototypeBuffers = skinnedInstance->GetPrototypeMesh()->buffers;
                const auto& skinnedBuffers = skinnedInstance->GetMesh()->buffers;

                nvrhi::BindingSetDesc setDesc;
                setDesc.bindings = {
                    nvrhi::BindingSetItem::PushConstants(0, sizeof(SkinningConstants)),
                    nvrhi::BindingSetItem::RawBuffer_SRV(0, prototypeBuffers->vertexBuffer),
                    nvrhi::BindingSetItem::RawBuffer_SRV(1, skinnedInstance->jointBuffer),
                    nvrhi::BindingSetItem::RawBuffer_UAV(0, skinnedBuffers->vertexBuffer)
                };

                skinnedInstance->skinningBindingSet =
                    m_Device->createBindingSet(setDesc, m_SkinningBindingLayout);
            }
        }

        MallocCUDABuffer(BindlessBuffers.size(), d_BindlessBuffers);
        cudaMemcpy(
            d_BindlessBuffers,
            BindlessBuffers.data(),
            BindlessBuffers.size() * sizeof(CUdeviceptr),
            cudaMemcpyHostToDevice);
    }

    nvrhi::BufferHandle OptiXScene::CreateMaterialBuffer()
    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = sizeof(MaterialConstants) * m_Resources->materialData.size();
        bufferDesc.debugName = "BindlessMaterials";
        bufferDesc.structStride = sizeof(MaterialConstants);
        bufferDesc.canHaveRawViews = true;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        bufferDesc.keepInitialState = true;
        bufferDesc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;

        auto buffer = m_Device->createBuffer(bufferDesc);
        t_MaterialConstants = reinterpret_cast<MaterialConstants*>(mapBufferToCUDABuffer(buffer, 0, m_Device));
         
        return buffer;
    }

    nvrhi::BufferHandle OptiXScene::CreateGeometryBuffer()
    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = sizeof(GeometryData) * m_Resources->geometryData.size();
        bufferDesc.debugName = "BindlessGeometry";
        bufferDesc.structStride = sizeof(GeometryData);
        bufferDesc.canHaveRawViews = true;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        bufferDesc.keepInitialState = true;
        bufferDesc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;

        auto buffer = m_Device->createBuffer(bufferDesc);
        t_GeometryData =
            reinterpret_cast<GeometryData*>(mapBufferToCUDABuffer(buffer, 0, m_Device));

        return buffer;
    }

    nvrhi::BufferHandle OptiXScene::CreateInstanceBuffer()
    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = sizeof(InstanceData) * m_Resources->instanceData.size();
        bufferDesc.debugName = "Instances";
        bufferDesc.structStride = m_EnableBindlessResources ? sizeof(InstanceData) : 0;
        bufferDesc.canHaveRawViews = true;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.isVertexBuffer = true;
        bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        bufferDesc.keepInitialState = true;
        bufferDesc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;

        auto buffer = m_Device->createBuffer(bufferDesc);
        t_InstanceData =
            reinterpret_cast<InstanceData*>(mapBufferToCUDABuffer(buffer, 0, m_Device));
        CUDA_SYNC_CHECK();

        return buffer;
    }

    nvrhi::BufferHandle OptiXScene::CreateMaterialConstantBuffer(const std::string& debugName)
    {
        return Scene::CreateMaterialConstantBuffer(debugName);
    }

    void OptiXScene::UpdateMaterial(const std::shared_ptr<Material>& material)
    {
        auto& materialData = m_Resources->materialData[material->materialID];
        if (m_EnableBindlessResources)
        {
            createTextureDescriptor(material->baseOrDiffuseTexture);
            createTextureDescriptor(material->metalRoughOrSpecularTexture);
            createTextureDescriptor(material->normalTexture);
            createTextureDescriptor(material->emissiveTexture);
            createTextureDescriptor(material->occlusionTexture);
            createTextureDescriptor(material->transmissionTexture);

            MallocCUDABuffer(BindlessTextures.size(), d_BindlessTextures);

            cudaMemcpy(
                d_BindlessTextures,
                BindlessTextures.data(),
                BindlessTextures.size() * sizeof(CUtexObject),
                cudaMemcpyHostToDevice);
        }
        material->FillConstantBuffer(materialData);
    }

    // remember to free blasDesc.vertexBuffer.
    nvrhi::OptiXTraversableHandle BuildMeshTraversableDesc(
        donut::engine::MeshInfo& mesh,
        nvrhi::IDevice* device)
    {
        nvrhi::OptiXTraversableHandle mesh_handle;

        auto mesh_count = mesh.geometries.size();
        nvrhi::OptiXTraversableDesc blasDesc;

        blasDesc.buildOptions = {};
        blasDesc.buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
        blasDesc.buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        blasDesc.buildOptions.motionOptions.numKeys = 1;

        blasDesc.buildInput = {};
        blasDesc.buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto& triangles = blasDesc.buildInput.triangleArray;
        CUdeviceptr vertex_buffer_ptr[1];
        unsigned triangle_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };

        std::vector<OptixInstance> instances(mesh_count);
        OptixInstance* d_instances = 0;
        MallocCUDABuffer(mesh_count, d_instances);
        nvrhi::OptiXTraversableDesc tlasDesc;
        for (int i = 0; i < mesh_count; ++i)
        {
            const auto& geometry = mesh.geometries[i];
            auto indexBuffer = mapBufferToCUDABuffer(mesh.buffers->indexBuffer, 0, device);
            triangles.indexBuffer =
                (CUdeviceptr)indexBuffer +
                (mesh.indexOffset + geometry->indexOffsetInMesh) * sizeof(uint32_t);
            mesh.buffers->indexBuffer;
            triangles.indexStrideInBytes = 3 * sizeof(uint32_t);
            triangles.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

            triangles.numIndexTriplets = geometry->numIndices / 3;

            vertex_buffer_ptr[0] =
                (CUdeviceptr)mapBufferToCUDABuffer(mesh.buffers->vertexBuffer, 0, device) +
                (mesh.vertexOffset + geometry->vertexOffsetInMesh) * sizeof(donut::math::float3) +
                mesh.buffers->getVertexBufferRange(donut::engine::VertexAttribute::Position)
                    .byteOffset;

            triangles.vertexBuffers = vertex_buffer_ptr;
            triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangles.vertexStrideInBytes = sizeof(donut::math::float3);
            triangles.numVertices = geometry->numVertices;
            triangles.numSbtRecords = 1;
            triangles.flags = triangle_flags;
            
            auto blas_handle = device->createOptiXTraversable(blasDesc);
            tlasDesc.handles.push_back(blas_handle);

            OptixInstance instance = {};
            instance.instanceId = i;
            donut::math::affineToColumnMajor(
                dm::translation(dm::float3{ 0, 0, 0 }),
                instance.transform);
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.sbtOffset = 0;
            instance.traversableHandle = blas_handle->getOptiXTraversable();
            instance.visibilityMask = 255;

            instances[i] = instance;
        }
        cudaMemcpy(
            d_instances,
            instances.data(),
            sizeof(OptixInstance) * instances.size(),
            cudaMemcpyHostToDevice);

        tlasDesc.buildInput = {};
        tlasDesc.buildOptions = blasDesc.buildOptions;
        tlasDesc.buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        tlasDesc.buildInput.instanceArray.numInstances = static_cast<unsigned>(mesh_count);
        tlasDesc.buildInput.instanceArray.instances = (CUdeviceptr)d_instances;
        tlasDesc.buildInput.instanceArray.instanceStride = sizeof(OptixInstance);

        mesh_handle = device->createOptiXTraversable(tlasDesc);
        cudaFree(d_instances);

        return mesh_handle;
    }

    nvrhi::OptiXTraversableHandle OptiXScene::buildOptixAccelStructs()
    {
        nvrhi::OptiXTraversableDesc tlasDesc;
        tlasDesc.buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        tlasDesc.buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
        tlasDesc.buildOptions.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
        tlasDesc.buildInput = {};
        tlasDesc.buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

        for (const auto& mesh : GetSceneGraph()->GetMeshes())
        {
            if (mesh->buffers->hasAttribute(VertexAttribute::JointWeights))
                continue; // skip the skinning prototypes

            auto as = BuildMeshTraversableDesc(*mesh, m_Device);
            tlasDesc.handles.push_back(as);
            mesh->optixAccel = as;
        }

        auto& instanceDesc = tlasDesc.buildInput.instanceArray;

        OptixInstance* d_instances = 0;

        auto numInstances = GetSceneGraph()->GetMeshInstances().size();
        std::vector<OptixInstance> instances(numInstances);

        MallocCUDABuffer(numInstances, d_instances);
        for (int i = 0; i < numInstances; ++i)
        {
            auto& instance = GetSceneGraph()->GetMeshInstances()[i];
            OptixInstance instanceDesc = {};

            instanceDesc.traversableHandle = instance->GetMesh()->optixAccel->getOptiXTraversable();
            assert(instanceDesc.traversableHandle);

            instanceDesc.visibilityMask = 255;
            instanceDesc.instanceId = instance->GetInstanceIndex();
            instanceDesc.flags = OPTIX_INSTANCE_FLAG_NONE;
            instanceDesc.sbtOffset = 0;

            auto node = instance->GetNode();
            assert(node);
            dm::affineToColumnMajor(node->GetLocalToWorldTransformFloat(), instanceDesc.transform);

            instances[i] = instanceDesc;
        }

        cudaMemcpy(
            d_instances,
            instances.data(),
            sizeof(OptixInstance) * numInstances,
            cudaMemcpyHostToDevice);

        instanceDesc.instances = (CUdeviceptr)d_instances;
        instanceDesc.numInstances = numInstances;

        handle = m_Device->createOptiXTraversable(tlasDesc);

        cudaFree(d_instances);

        return handle;
    }
} // namespace donut::engine
