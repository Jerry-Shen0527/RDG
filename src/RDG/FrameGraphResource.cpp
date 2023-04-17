#include "RDG/RDG/FrameGraphResource.h"

#include "PassNode.h"
#include "RDG/RDG/FrameGraph.h"

namespace Furnace
{
    FrameGraphResources::RenderPassInfo FrameGraphResources::getRenderPassInfo(uint32_t id) const
    {
        // this cast is safe because this can only be called from a RenderPassNode
        const auto& renderPassNode = static_cast<const RenderPassNode&>(mPassNode);
        const RenderPassNode::RenderPassData* pRenderPassData =
            renderPassNode.getRenderPassData(id);
        return { pRenderPassData->command_list, pRenderPassData->state };
    }

    FrameGraphResources::ComputePassInfo FrameGraphResources::getComputePassInfo() const
    {
        // this cast is safe because this can only be called from a RenderPassNode
        const auto& computePassNode = static_cast<const ComputePassNode&>(mPassNode);
        const ComputePassNode::ComputePassData* pComputePassData =
            computePassNode.getComputePassData();
        return { pComputePassData->command_list, pComputePassData->state };
    }

    FrameGraphResources::RayTracingPassInfo FrameGraphResources::getRayTracingPassInfo() const
    {
        // this cast is safe because this can only be called from a RenderPassNode
        const auto& computePassNode = static_cast<const RayTracingPassNode&>(mPassNode);
        const RayTracingPassNode::RayTracingPassData* pComputePassData =
            computePassNode.getRayTracingPassData();
        return { pComputePassData->command_list, pComputePassData->state };
    }

#ifdef RDG_WITH_CUDA
    FrameGraphResources::CudaPassInfo FrameGraphResources::getCudaPassInfo() const
    {
        // this cast is safe because this can only be called from a RenderPassNode
        const auto& renderPassNode = static_cast<const CudaPassNode&>(mPassNode);
        const CudaPassNode::CudaPassData* pCudaPassData = renderPassNode.getCudaPassData();
        return { };
    }
#endif

#ifdef RDG_WITH_OPTIX
    FrameGraphResources::OptiXPassInfo FrameGraphResources::getOptiXPassInfo() const
    {
        // this cast is safe because this can only be called from a RenderPassNode
        const auto& renderPassNode = static_cast<const OptiXPassNode&>(mPassNode);
        const OptiXPassNode::OptiXPassData* pOptiXPassData = renderPassNode.getOptiXPassData();
        return { pOptiXPassData->solid_handle, pOptiXPassData->sbt };
    }
#endif

    VirtualResource& FrameGraphResources::getResource(FrameGraphHandle handle) const
    {
        VirtualResource* const resource = mFrameGraph.getResource(handle);
        auto& declaredHandles = mPassNode.mDeclaredHandles;
        const bool hasReadOrWrite =
            declaredHandles.find(handle.index) != declaredHandles.cend();

        if (!hasReadOrWrite)
        {
            printf(
                "Pass \"%s\" didn't declare any access to resource \"%s\"",
                mPassNode.getName(),
                resource->name);
            assert(0);
        }


        return *resource;
    }
}
