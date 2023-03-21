#include"RDG/RenderUtils.h"

#include "donut/engine/Scene.h"
#include "donut/engine/SceneTypes.h"
#include"nvrhi/nvrhi.h"
#include "nvrhi/utils.h"

namespace Furnace
{
    void GetMeshBlasDesc(donut::engine::MeshInfo& mesh, nvrhi::rt::AccelStructDesc& blasDesc)
    {
        blasDesc.isTopLevel = false;
        blasDesc.debugName = mesh.name;

        for (const auto& geometry : mesh.geometries)
        {
            nvrhi::rt::GeometryDesc geometryDesc;
            auto& triangles = geometryDesc.geometryData.triangles;
            triangles.indexBuffer = mesh.buffers->indexBuffer;
            triangles.indexOffset =
                (mesh.indexOffset + geometry->indexOffsetInMesh) * sizeof(uint32_t);
            triangles.indexFormat = nvrhi::Format::R32_UINT;
            triangles.indexCount = geometry->numIndices;
            triangles.vertexBuffer = mesh.buffers->vertexBuffer;
            triangles.vertexOffset =
                (mesh.vertexOffset + geometry->vertexOffsetInMesh) * sizeof(donut::math::float3) +
                mesh.buffers->getVertexBufferRange(donut::engine::VertexAttribute::Position)
                    .byteOffset;
            triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
            triangles.vertexStride = sizeof(donut::math::float3);
            triangles.vertexCount = geometry->numVertices;
            geometryDesc.geometryType = nvrhi::rt::GeometryType::Triangles;
            geometryDesc.flags =
                (geometry->material->domain == donut::engine::MaterialDomain::AlphaTested)
                    ? nvrhi::rt::GeometryFlags::None
                    : nvrhi::rt::GeometryFlags::Opaque;
            blasDesc.bottomLevelGeometries.push_back(geometryDesc);
        }

        // don't compact acceleration structures that are built per frame
        if (mesh.skinPrototype != nullptr)
        {
            blasDesc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace;
        }
        else
        {
            blasDesc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace |
                                  nvrhi::rt::AccelStructBuildFlags::AllowCompaction;
        }
    }

    nvrhi::rt::AccelStructHandle CreateAccelStructs(
        std::shared_ptr<donut::engine::Scene> m_Scene,
        nvrhi::DeviceHandle device,
        nvrhi::CommandListHandle commandList)
    {
        nvrhi::rt::AccelStructHandle handle;

        for (const auto& mesh : m_Scene->GetSceneGraph()->GetMeshes())
        {
            if (mesh->buffers->hasAttribute(donut::engine::VertexAttribute::JointWeights))
                continue; // skip the skinning prototypes

            nvrhi::rt::AccelStructDesc blasDesc;

            GetMeshBlasDesc(*mesh, blasDesc);

            nvrhi::rt::AccelStructHandle as = device->createAccelStruct(blasDesc);

            if (!mesh->skinPrototype)
                nvrhi::utils::BuildBottomLevelAccelStruct(commandList, as, blasDesc);

            mesh->accelStruct = as;
        }

        nvrhi::rt::AccelStructDesc tlasDesc;
        tlasDesc.isTopLevel = true;
        tlasDesc.topLevelMaxInstances = m_Scene->GetSceneGraph()->GetMeshInstances().size();
        handle = device->createAccelStruct(tlasDesc);

        return handle;
    }


    void BuildTLAS(
        nvrhi::rt::AccelStructHandle handle,
        std::shared_ptr<donut::engine::Scene> m_Scene,
        nvrhi::CommandListHandle commandList,
        uint32_t frameIndex)
    {
        commandList->beginMarker("Skinned BLAS Updates");

        // Transition all the buffers to their necessary states before building the BLAS'es to allow
        // BLAS batching
        for (const auto& skinnedInstance : m_Scene->GetSceneGraph()->GetSkinnedMeshInstances())
        {
            if (skinnedInstance->GetLastUpdateFrameIndex() < frameIndex)
                continue;

            commandList->setAccelStructState(
                skinnedInstance->GetMesh()->accelStruct,
                nvrhi::ResourceStates::AccelStructWrite);
            commandList->setBufferState(
                skinnedInstance->GetMesh()->buffers->vertexBuffer,
                nvrhi::ResourceStates::AccelStructBuildInput);
        }
        commandList->commitBarriers();

        // Now build the BLAS'es
        for (const auto& skinnedInstance : m_Scene->GetSceneGraph()->GetSkinnedMeshInstances())
        {
            if (skinnedInstance->GetLastUpdateFrameIndex() < frameIndex)
                continue;

            nvrhi::rt::AccelStructDesc blasDesc;
            GetMeshBlasDesc(*skinnedInstance->GetMesh(), blasDesc);

            nvrhi::utils::BuildBottomLevelAccelStruct(
                commandList,
                skinnedInstance->GetMesh()->accelStruct,
                blasDesc);
        }
        commandList->endMarker();

        std::vector<nvrhi::rt::InstanceDesc> instances;

        for (const auto& instance : m_Scene->GetSceneGraph()->GetMeshInstances())
        {
            nvrhi::rt::InstanceDesc instanceDesc;
            instanceDesc.bottomLevelAS = instance->GetMesh()->accelStruct;
            assert(instanceDesc.bottomLevelAS);
            instanceDesc.instanceMask = 1;
            instanceDesc.instanceID = instance->GetInstanceIndex();

            auto node = instance->GetNode();
            assert(node);
            dm::affineToColumnMajor(node->GetLocalToWorldTransformFloat(), instanceDesc.transform);

            instances.push_back(instanceDesc);
        }

        // Compact acceleration structures that are tagged for compaction and have finished
        // executing the original build
        commandList->compactBottomLevelAccelStructs();

        commandList->beginMarker("TLAS Update");
        commandList->buildTopLevelAccelStruct(handle, instances.data(), instances.size());
        commandList->endMarker();

    }
} // namespace Furnace
