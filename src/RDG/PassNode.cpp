#include"PassNode.h"

#include "RDG/RDG/FrameGraph.h"

#include <RDG/RDG/DescTrait.h>

namespace Furnace
{
    PassNode::PassNode(FrameGraph& fg)
        : Node(fg.getDependencyGraph()),
          mFrameGraph(fg)
    {
    }

    PassNode::PassNode(PassNode&& rhs) noexcept = default;

    void PassNode::registerResource(FrameGraphHandle resourceHandle) noexcept
    {
        VirtualResource* resource = mFrameGraph.getResource(resourceHandle);
        resource->neededByPass(this);
        mDeclaredHandles.insert(resourceHandle.index);
    }

    void RenderPassNode::RenderPassData::devirtualize(
        FrameGraph& fg,
        ResourceAllocator& resourceAllocator) noexcept
    {
        if (!imported)
        {
            descriptor.resolve(resourceAllocator);

            auto subresources = descriptor.viewport->GetSubresources();
            nvrhi::FramebufferDesc framebuffer_desc;
            for (size_t i = 0; i < nvrhi::c_MaxRenderTargets; i++)
            {
                if (attachmentInfo[i])
                {
                    const auto* pResource = static_cast<const Resource<nvrhi::TextureHandle>*>(
                        fg.getResource(attachmentInfo[i]));
                    auto handle = pResource->resource;

                    framebuffer_desc.addColorAttachment(handle, subresources);
                }
            }

            for (size_t i = 0; i < 1; i++)
            {
                if (attachmentInfo[nvrhi::c_MaxRenderTargets + i])
                {
                    const auto* pResource = static_cast<const Resource<nvrhi::TextureHandle>*>(
                        fg.getResource(attachmentInfo[nvrhi::c_MaxRenderTargets + i]));
                    auto handle = pResource->resource;

                    framebuffer_desc.setDepthAttachment(handle, subresources);
                }
            }

            state.framebuffer = resourceAllocator.create(framebuffer_desc);

            state.pipeline = resourceAllocator.create(descriptor.pipelineDesc, state.framebuffer);
            //state.bindings = descriptor.binding_layouts;
            state.setViewport(descriptor.viewport->GetViewportState());

            command_list = resourceAllocator.create(nvrhi::CommandListDesc{});

            command_list->open();

            for (uint32_t i = 0; i < framebuffer_desc.colorAttachments.size(); ++i)
            {
                if ((TargetBufferFlags(i+1) & descriptor.clearFlags) != 0)
                {
                    command_list->clearTextureFloat(
                        framebuffer_desc.colorAttachments[i].texture,
                        nvrhi::AllSubresources,
                        descriptor.clearColor);
                }
            }
            if ((TargetBufferFlags::DEPTH & descriptor.clearFlags) != 0)
            {
                nvrhi::TextureHandle Depth = framebuffer_desc.depthAttachment.texture;
                const nvrhi::FormatInfo& depthFormatInfo =
                    nvrhi::getFormatInfo(Depth->getDesc().format);

                command_list->clearDepthStencilTexture(
                    Depth,
                    nvrhi::AllSubresources,
                    true,
                    descriptor.clearDepth,
                    depthFormatInfo.hasStencil,
                    0);
            }
            command_list->close();

            resourceAllocator.GetNvrhiDevice()->executeCommandList(command_list);
        }
    }

    void RenderPassNode::RenderPassData::destroy(ResourceAllocator& resourceAllocator) noexcept
    {
        if ((!imported))
        {
            resourceAllocator.destroy(state.framebuffer);
            resourceAllocator.destroy(state.pipeline);
            resourceAllocator.destroy(command_list);
        }
    }

    const RenderPassNode::RenderPassData* RenderPassNode::getRenderPassData(
        uint32_t id) const noexcept
    {
        return id < mRenderTargetData.size() ? &mRenderTargetData[id] : nullptr;
    }

    void RenderPassNode::execute(
        const FrameGraphResources& resources,
        nvrhi::DeviceHandle& device) noexcept
    {
        FrameGraph& fg = mFrameGraph;
        // ResourceAllocatorInterface& resourceAllocator = fg.getResourceAllocator();

        // create the render targets
        for (auto& rt : mRenderTargetData)
        {
            rt.devirtualize(fg, fg.getResourceAllocator());
        }

        mPassBase->execute(resources, device);

        // destroy the render targets
        for (auto& rt : mRenderTargetData)
        {
            rt.destroy(fg.getResourceAllocator());
        }
    }

    uint32_t RenderPassNode::declareRenderTarget(
        FrameGraph& fg,
        FrameGraph::Builder& builder,
        const char* name,
        const FrameGraphRenderPass::Descriptor& descriptor)
    {
        RenderPassData data;
        data.name = name;
        data.descriptor = descriptor;
        FrameGraphRenderPass::Attachments& attachments = data.descriptor.attachments;

        // retrieve the ResourceNode of the attachments coming to us -- this will be used later
        // to compute the discard flags.

        const DependencyGraph& dependencyGraph = fg.getDependencyGraph();
        auto incomingEdges = dependencyGraph.getIncomingEdges(this);
        auto outgoingEdges = dependencyGraph.getOutgoingEdges(this);

        for (size_t i = 0; i < RenderPassData::ATTACHMENT_COUNT; i++)
        {
            if (descriptor.attachments.array[i])
            {
                data.attachmentInfo[i] = attachments.array[i];

                // TODO: this is not very efficient
                auto incomingPos = std::find_if(
                    incomingEdges.begin(),
                    incomingEdges.end(),
                    [&dependencyGraph,
                        handle = descriptor.attachments.array[i]](const DependencyGraph::Edge* edge)
                    {
                        auto node =
                            static_cast<const ResourceNode*>(dependencyGraph.getNode(edge->from));
                        return node->resourceHandle == handle;
                    });

                if (incomingPos != incomingEdges.end())
                {
                    data.incoming[i] = const_cast<ResourceNode*>(static_cast<const ResourceNode*>(
                        dependencyGraph.getNode((*incomingPos)->from)));
                }

                // this could be either outgoing or incoming (if there are no outgoing)
                data.outgoing[i] = fg.getActiveResourceNode(descriptor.attachments.array[i]);
                if (data.outgoing[i] == data.incoming[i])
                {
                    data.outgoing[i] = nullptr;
                }
            }
        }

        uint32_t id = mRenderTargetData.size();
        mRenderTargetData.push_back(data);
        return id;
    }

    void RenderPassNode::resolve() noexcept
    {
        // TODO: Really use nvrhi to do the settings!
    }


    PresentPassNode::PresentPassNode(FrameGraph& fg) noexcept
        : PassNode(fg)
    {
    }

    PresentPassNode::PresentPassNode(PresentPassNode&& rhs) noexcept = default;
    PresentPassNode::~PresentPassNode() noexcept = default;

    const char* PresentPassNode::getName() const noexcept
    {
        return "Present";
    }

    void AccumulatePassNode::execute(
        const FrameGraphResources& resources,
        nvrhi::DeviceHandle& driver) noexcept
    {
        
    }

    void AccumulatePassNode::resolve() noexcept
    {
    }

    void ComputePassNode::ComputePassData::devirtualize(
        FrameGraph& fg,
        ResourceAllocator& resourceAllocator) noexcept
    {
        {
            descriptor.resolve(resourceAllocator);
            state.pipeline = resourceAllocator.create(descriptor.pipelineDesc);
            command_list = resourceAllocator.create(nvrhi::CommandListDesc{});
        }
    }

    void ComputePassNode::ComputePassData::destroy(ResourceAllocator& resourceAllocator) noexcept
    {
        resourceAllocator.destroy(state.pipeline);
        resourceAllocator.destroy(command_list);
    }

    void ComputePassNode::declareComputePass(
        FrameGraph& fg,
        FrameGraph::Builder& builder,
        const char* name,
        const FrameGraphComputePass::Descriptor& descriptor)
    {
        mComputeData.name = name;
        mComputeData.descriptor = descriptor;
    }

    const ComputePassNode::ComputePassData* ComputePassNode::getComputePassData() const noexcept
    {
        return &mComputeData;
    }

    void RayTracingPassNode::RayTracingPassData::devirtualize(
        FrameGraph& fg,
        ResourceAllocator& resourceAllocator) noexcept
    {
        {
            descriptor.resolve(resourceAllocator);
            pipeline = resourceAllocator.create(descriptor.pipelineDesc);
            state.shaderTable = resourceAllocator.create(pipeline);
            state.shaderTable->clearCallableShaders();
            state.shaderTable->clearHitShaders();
            state.shaderTable->clearMissShaders();
            command_list = resourceAllocator.create(nvrhi::CommandListDesc{});
        }
    }

    void RayTracingPassNode::RayTracingPassData::destroy(
        ResourceAllocator& resourceAllocator) noexcept
    {
        resourceAllocator.destroy(state.shaderTable);
        resourceAllocator.destroy(pipeline);
        resourceAllocator.destroy(command_list);
    }

    void RayTracingPassNode::declareRayTracingPass(
        FrameGraph& fg,
        FrameGraph::Builder& builder,
        const char* name,
        const FrameGraphRayTracingPass::Descriptor& descriptor)
    {
        mRayTracingData.name = name;
        mRayTracingData.descriptor = descriptor;
    }

    const RayTracingPassNode::RayTracingPassData* RayTracingPassNode::
    getRayTracingPassData() const noexcept
    {
        return &mRayTracingData;
    }

    void PresentPassNode::execute(const FrameGraphResources&, nvrhi::DeviceHandle&) noexcept
    {
    }

    void PresentPassNode::resolve() noexcept
    {
    }
}
