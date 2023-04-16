#pragma once
#include <memory>
#include <unordered_set>

#include "nvrhi/nvrhi.h"
#include "RDG/RDG/DependencyGraph.h"
#include "RDG/RDG/FrameGraph.h"
#include "RDG/RDG/FrameGraphId.h"

namespace Furnace
{
    class FrameGraphPassBase;
    class FrameGraphHandle;
    class FrameGraphResources;
    class FrameGraph;
    class VirtualResource;

    class PassNode : public DependencyGraph::Node
    {
        friend class FrameGraphResources;
    public:
        explicit PassNode(FrameGraph& fg);
        PassNode(PassNode&& rhs) noexcept;
        PassNode(const PassNode&) = delete;
        PassNode& operator=(const PassNode&) = delete;

        virtual void execute(
            const FrameGraphResources& resources,
            nvrhi::DeviceHandle& driver) noexcept = 0;

        ~PassNode() override
        {
        }

        void registerResource(FrameGraphHandle resource_handle) noexcept;
        virtual void resolve() noexcept = 0;
        const char* mName = nullptr;

        const char* getName() const noexcept override
        {
            return mName;
        }

        std::vector<VirtualResource*> devirtualize;
        std::vector<VirtualResource*> destroy;
    protected:
        /**
         * \brief Records which resources are being used by this pass, in registerResource()
         */
        std::unordered_set<FrameGraphHandle::Index> mDeclaredHandles;
        FrameGraph& mFrameGraph;
    };

    // A renderpass always has a render target, right?
    class RenderPassNode : public PassNode
    {
    public:
        class RenderPassData
        {
        public:
            static constexpr size_t ATTACHMENT_COUNT = nvrhi::c_MaxRenderTargets + 1;
            const char* name = {};
            bool imported = false;

            FrameGraphId<nvrhi::TextureHandle> attachmentInfo[ATTACHMENT_COUNT] = {};

            ResourceNode* incoming[ATTACHMENT_COUNT] = {}; // nodes of the incoming attachments
            ResourceNode* outgoing[ATTACHMENT_COUNT] = {}; // nodes of the outgoing attachments


            FrameGraphRenderPass::Descriptor descriptor;
            nvrhi::CommandListHandle command_list;
            nvrhi::GraphicsState state;

            void devirtualize(FrameGraph& fg, ResourceAllocator& resourceAllocator) noexcept;

            void destroy(ResourceAllocator& resourceAllocator) noexcept;
        };

        RenderPassNode(FrameGraph& fg, const char* name, FrameGraphPassBase* base) noexcept
            : PassNode(fg),
              mPassBase(base),
              mName(name)
        {
        }

        uint32_t declareRenderTarget(
            FrameGraph& fg,
            FrameGraph::Builder& builder,
            const char* name,
            const FrameGraphRenderPass::Descriptor& descriptor);

        std::unique_ptr<FrameGraphPassBase> mPassBase;

        const RenderPassData* getRenderPassData(uint32_t id) const noexcept;

    private:
        void execute(const FrameGraphResources& resources, nvrhi::DeviceHandle& device) noexcept
        override;
        const char* const mName;

        void resolve() noexcept override;
        std::vector<RenderPassData> mRenderTargetData;
    };

    class PresentPassNode : public PassNode
    {
    public:
        explicit PresentPassNode(FrameGraph& fg) noexcept;
        PresentPassNode(PresentPassNode&& rhs) noexcept;
        ~PresentPassNode() noexcept override;
        PresentPassNode(const PresentPassNode&) = delete;
        PresentPassNode& operator=(const PresentPassNode&) = delete;
        void resolve() noexcept override;
        void execute(const FrameGraphResources& resources, nvrhi::DeviceHandle& driver) noexcept
        override;

    private:
        // virtuals from DependencyGraph::Node
        const char* getName() const noexcept override;
    };

    struct AccumulateParameters
    {
        
    };

    class AccumulatePassNode :public PassNode
    {
    public:
        explicit AccumulatePassNode(FrameGraph& fg,const AccumulateParameters& parameters)
            : PassNode(fg)
        {
        }

        AccumulatePassNode(AccumulatePassNode&& rhs) noexcept = default;
        AccumulatePassNode(const AccumulatePassNode& pass_node) noexcept = default;

        void execute(
            const FrameGraphResources& resources,
            nvrhi::DeviceHandle& driver) noexcept override;
        void resolve() noexcept override;
    };

    // A renderpass always has a render target, right?
    class ComputePassNode : public PassNode
    {
    public:
        class ComputePassData
        {
        public:
            const char* name = {};

            FrameGraphComputePass::Descriptor descriptor;
            nvrhi::CommandListHandle command_list;
            nvrhi::ComputeState state;

            void devirtualize(FrameGraph& fg, ResourceAllocator& resourceAllocator) noexcept;

            void destroy(ResourceAllocator& resourceAllocator) noexcept;
        };

        ComputePassNode(FrameGraph& fg, const char* name, FrameGraphPassBase* base) noexcept
            : PassNode(fg),
              mPassBase(base),
              mName(name)
        {
        }

        void declareComputePass(
            FrameGraph& fg,
            FrameGraph::Builder& builder,
            const char* name,
            const FrameGraphComputePass::Descriptor& descriptor);

        std::unique_ptr<FrameGraphPassBase> mPassBase;

        const ComputePassData* getComputePassData() const noexcept;

    private:
        void execute(const FrameGraphResources& resources, nvrhi::DeviceHandle& device) noexcept
        override
        {
            FrameGraph& fg = mFrameGraph;

            mComputeData.devirtualize(fg, fg.getResourceAllocator());
            mPassBase->execute(resources, device);
            mComputeData.destroy(fg.getResourceAllocator());
        }

        const char* const mName;

        void resolve() noexcept override
        {
        }

        ComputePassData mComputeData;
    };

        // A renderpass always has a render target, right?
    class RayTracingPassNode : public PassNode
    {
       public:
        class RayTracingPassData
        {
           public:
            const char* name = {};

            FrameGraphRayTracingPass::Descriptor descriptor;
            nvrhi::CommandListHandle command_list;


            nvrhi::rt::State state;
            nvrhi::rt::PipelineHandle pipeline;

            void devirtualize(FrameGraph& fg, ResourceAllocator& resourceAllocator) noexcept;

            void destroy(ResourceAllocator& resourceAllocator) noexcept;
        };

        RayTracingPassNode(FrameGraph& fg, const char* name, FrameGraphPassBase* base) noexcept
            : PassNode(fg),
              mPassBase(base),
              mName(name)
        {
        }

        void declareRayTracingPass(
            FrameGraph& fg,
            FrameGraph::Builder& builder,
            const char* name,
            const FrameGraphRayTracingPass::Descriptor& descriptor);

        std::unique_ptr<FrameGraphPassBase> mPassBase;

        const RayTracingPassData* getRayTracingPassData() const noexcept;

       private:
        void execute(const FrameGraphResources& resources, nvrhi::DeviceHandle& device) noexcept
            override
        {
            FrameGraph& fg = mFrameGraph;

            mRayTracingData.devirtualize(fg, fg.getResourceAllocator());
            mPassBase->execute(resources, device);
            mRayTracingData.destroy(fg.getResourceAllocator());
        }

        const char* const mName;

        void resolve() noexcept override
        {
        }

        RayTracingPassData mRayTracingData;
    };


    
#ifdef RDG_WITH_CUDA
    class CudaPassNode : public PassNode
    {
       public:
        class CudaPassData
        {
           public:
            const char* name = {};
            FrameGraphCudaPass::Descriptor descriptor;
        };

        CudaPassNode(FrameGraph& fg, const char* name, FrameGraphPassBase* base) noexcept
            : PassNode(fg),
              mPassBase(base),
              mName(name)
        {
        }

        std::unique_ptr<FrameGraphPassBase> mPassBase;

        void declareCudaPass(
            FrameGraph& fg,
            FrameGraph::Builder& builder,
            const char* name,
            const FrameGraphCudaPass::Descriptor& descriptor)
        {
            mCudaData.name = name;
            mCudaData.descriptor = descriptor;
        }

        const CudaPassData* getCudaPassData() const noexcept
        {
            return &mCudaData;
        }

    private:
        void execute(const FrameGraphResources& resources, nvrhi::DeviceHandle& device) noexcept
        override
        {
            mPassBase->execute(resources, device);
        }

        const char* const mName;

        void resolve() noexcept override
        {
        }

        CudaPassData mCudaData;
    };
#endif

#ifdef RDG_WITH_OPTIX

    class OptiXPassNode : public PassNode
    {
       public:
        class OptiXPassData
        {
           public:
            const char* name = {};

            FrameGraphOptiXPass::Descriptor descriptor;

            void devirtualize(FrameGraph& fg, ResourceAllocator& resourceAllocator) noexcept;

            void destroy(ResourceAllocator& resourceAllocator) noexcept;

            nvrhi::OptiXPipelineHandle solid_handle;
            OptixShaderBindingTable sbt = {};

           private:
            std::vector<nvrhi::OptiXModuleHandle> solidModules;
            nvrhi::OptiXProgramGroupHandle solid_raygen_group;
            std::vector<nvrhi::OptiXProgramGroupHandle> solid_hitgroup_group;
            std::vector<nvrhi::OptiXProgramGroupHandle> solid_miss_group;

            nvrhi::CudaLinearBufferHandle raygen_record;
            nvrhi::CudaLinearBufferHandle hitgroup_record;
            nvrhi::CudaLinearBufferHandle miss_record;
        };

        OptiXPassNode(FrameGraph& fg, const char* name, FrameGraphPassBase* base) noexcept
            : PassNode(fg),
              mPassBase(base),
              mName(name)
        {
        }

        std::unique_ptr<FrameGraphPassBase> mPassBase;

        void declareOptiXPass(
            FrameGraph& fg,
            FrameGraph::Builder& builder,
            const char* name,
            const FrameGraphOptiXPass::Descriptor& descriptor);

        const OptiXPassData* getOptiXPassData() const noexcept;

       private:
        void execute(const FrameGraphResources& resources, nvrhi::DeviceHandle& device) noexcept
            override;

        const char* const mName;

        void resolve() noexcept override;
        OptiXPassData mOptiXData;
    };
#endif

}
