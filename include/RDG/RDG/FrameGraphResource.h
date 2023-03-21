#pragma once
#include "FrameGraphId.h"
#include "Resource.h"

namespace Furnace
{
    class PassNode;
    class FrameGraph;

    class FrameGraphResources
    {
    public:
        FrameGraphResources(FrameGraph& fg, PassNode& passNode) noexcept
            : mFrameGraph(fg),
              mPassNode(passNode)
        {
        }

        FrameGraphResources(const FrameGraphResources&) = delete;
        FrameGraphResources& operator=(const FrameGraphResources&) = delete;

        struct RenderPassInfo
        {
            nvrhi::CommandListHandle command_list;
            nvrhi::GraphicsState state;
        };

        struct ComputePassInfo
        {
            nvrhi::CommandListHandle command_list;
            nvrhi::ComputeState state;
        };

        struct RayTracingPassInfo
        {
            nvrhi::CommandListHandle command_list;
            nvrhi::rt::State state;
        };


        template<typename RESOURCE, typename ... Args>
        const RESOURCE& get(FrameGraphId<RESOURCE> handle, Args&&... args) const
        {
            return static_cast<const Resource<RESOURCE, Args...>&>(getResource(handle)).resource;
        }


        // Use this to mannually restrain a resource from destroyed
        template<typename RESOURCE>
        void detach(
            FrameGraphId<RESOURCE> handle,
            RESOURCE* pOutResource,
            typename RESOURCE::Descriptor* pOutDescriptor) const
        {
            Resource<RESOURCE>& concrete = static_cast<Resource<RESOURCE>&>(getResource(handle));
            concrete.detached = true;
            assert(pOutResource);
            *pOutResource = concrete.resource;
            if (pOutDescriptor)
            {
                *pOutDescriptor = concrete.descriptor;
            }
        }

        RenderPassInfo getRenderPassInfo(uint32_t id = 0u) const;
        ComputePassInfo getComputePassInfo() const;
        RayTracingPassInfo getRayTracingPassInfo() const;

    private:
        VirtualResource& getResource(FrameGraphHandle handle) const;


        FrameGraph& mFrameGraph;
        PassNode& mPassNode;
    };
} // namespace Furnace
