#pragma once
#include <functional>
#include <iostream>
#include <vector>

#include "DependencyGraph.h"
#include "DescTrait.h"
#include "FrameGraphId.h"
#include "FrameGraphPass.h"
#include "Resource.h"
#include "ResourceNode.h"
#include "FrameGraphResource.h"

#include "nvrhi/nvrhi.h"


namespace Furnace
{
    class FrameGraphResources;

    class FrameGraphPassExecutor
    {
        friend class FrameGraph;
        friend class PassNode;

    protected:
        virtual void execute(
            const FrameGraphResources& resources,
            nvrhi::DeviceHandle& device) noexcept = 0;

    public:
        FrameGraphPassExecutor() noexcept = default;
        virtual ~FrameGraphPassExecutor() noexcept = default;
        FrameGraphPassExecutor(const FrameGraphPassExecutor&) = delete;
        FrameGraphPassExecutor& operator=(const FrameGraphPassExecutor&) = delete;
    };

    class FrameGraphPassBase : protected FrameGraphPassExecutor
    {
        friend class FrameGraph;
        friend class PassNode;
        friend class RenderPassNode;
        friend class ComputePassNode;
        friend class CudaPassNode;
        friend class RayTracingPassNode;

        PassNode* mNode = nullptr;

        void setNode(PassNode* node) noexcept
        {
            mNode = node;
        }

        const PassNode& getNode() const noexcept
        {
            return *mNode;
        }

    public:
        //using FrameGraphPassExecutor::FrameGraphPassExecutor;
        ~FrameGraphPassBase() noexcept override;
    };


    template<typename Data>
    class FrameGraphPass : public FrameGraphPassBase
    {
        friend class FrameGraph;

        //// allow our allocators to instantiate us
        //template<typename, typename, typename, typename>
        //friend class utils::Arena;

    protected:
        FrameGraphPass() = default;
        Data mData;

    public:
        const Data& getData() const noexcept
        {
            return mData;
        }

        const Data* operator->() const
        {
            return &mData;
        }

    protected:
        void execute(
            const FrameGraphResources& resources,
            nvrhi::DeviceHandle& device) noexcept override
        {
        }
    };


    /**
     * \brief FrameGraphPassConcrete contains both data and execution lambda closure, while FrameGraphPass contains only data. 
     * \tparam Data 
     * \tparam Execute 
     */
    template<typename Data, typename Execute>
    class FrameGraphPassConcrete : public FrameGraphPass<Data>
    {
    public:
        friend class FrameGraph;

        explicit FrameGraphPassConcrete(Execute&& execute) noexcept
            : mExecute(std::move(execute))
        {
        }


    protected:
        void execute(
            const FrameGraphResources& resources,
            nvrhi::DeviceHandle& device) noexcept override
        {
            mExecute(resources, this->mData, device);
        }

    public:
        Execute mExecute;
    };

    class FrameGraph
    {
    public:
        friend class ResourceNode;
        friend class RenderPassNode;
        friend class ComputePassNode;
        friend class RayTracingPassNode;

        enum class PassType
        {
            Render,
            Compute,
            RayTracing
        };

        class Builder
        {
        public:
            Builder(const Builder&) = delete;
            Builder& operator=(const Builder&) = delete;


            Builder(FrameGraph& fg, PassNode* node) noexcept;
            FrameGraph& mFrameGraph;
            PassNode* mPassNode;

            template<typename DESC, typename RESOURCE = resc<DESC>>
            FrameGraphId<RESOURCE> create(const char* name, const DESC& desc = {}) noexcept
            {
                return mFrameGraph.create<RESOURCE>(name, desc);
            }

            template<typename DESC, typename RESOURCE = resc<DESC>, typename... Args>
            FrameGraphId<RESOURCE> create(const DESC& desc, Args ... rest) noexcept
            {
                std::vector<FrameGraphHandle> parents;

                auto ret = createParentResourceFromDesc(parents, rest...);

                auto new_resource = std::apply(
                    [this, &desc]<typename... T>(T ... args)
                    {
                        return mFrameGraph.create<RESOURCE, T...>(
                            "",
                            desc,
                            std::forward<T>(args)...);
                    },
                    ret);
                mFrameGraph.getActiveResourceNode(new_resource)->setParents(parents);

                return new_resource;
            }

            template<typename RESOURCE>
            FrameGraphId<RESOURCE> import(const RESOURCE& resource) noexcept
            {
                return mFrameGraph.import<RESOURCE>("", resource);
            }


            template<typename RESOURCE>
            FrameGraphId<RESOURCE> read(
                FrameGraphId<RESOURCE> input,
                Usage usage = Usage::DEFAULT_R_USAGE)
            {
                return mFrameGraph.read<RESOURCE>(mPassNode, input, usage);
            }

            template<typename RESOURCE>
            [[nodiscard]] FrameGraphId<RESOURCE> write(
                FrameGraphId<RESOURCE> input,
                Usage usage = Usage::DEFAULT_W_USAGE)
            {
                return mFrameGraph.write<RESOURCE>(mPassNode, input, usage);
            }

            void sideEffect() noexcept;


            //Possibly use templates to resolve this.
            uint32_t declareRenderPass(
                const char* name,
                const FrameGraphRenderPass::Descriptor& desc);

            void declareComputePass(
                const char* name,
                const FrameGraphComputePass::Descriptor& desc);

            void declareRayTracingPass(
                const char* name,
                const FrameGraphRayTracingPass::Descriptor& desc);

        private:
            template<typename...Args>
            auto createParentResourceFromDesc(
                std::vector<FrameGraphHandle>& parents,
                Args&&... args)
            {
                return convertToSolid_impl(parents, std::tuple<>(), args...);
            }

            template<typename...T1, typename DESC, typename... Args>
            auto convertToSolid_impl(
                std::vector<FrameGraphHandle>& parents,
                std::tuple<T1...>&& tuple,
                const DESC& desc,
                Args&&... args)
            {
                auto created = create(desc);
                parents.push_back(created);

                auto resource = static_cast<Resource<resc<DESC>>*>(mFrameGraph.
                    getResource(created));
                auto processed = std::tuple_cat(tuple, std::make_tuple(resource));
                return convertToSolid_impl(parents, std::move(processed), args...);
            }

            template<typename... T1, typename RESOURCE, typename... Args>
            auto convertToSolid_impl(
                std::vector<FrameGraphHandle>& parents,
                std::tuple<T1...>&& tuple,
                const FrameGraphId<RESOURCE>& imported,
                Args&&... args)
            {
                parents.push_back(imported);
                auto resource = static_cast<Resource<RESOURCE>*>(mFrameGraph.getResource(imported));
                auto processed = std::tuple_cat(tuple, std::make_tuple(resource));
                return convertToSolid_impl(parents, std::move(processed), args...);
            }

            template<typename... T1>
            auto convertToSolid_impl(
                std::vector<FrameGraphHandle>& parents,
                std::tuple<T1...>&& tuple)
            {
                return tuple;
            }
        };


        ~FrameGraph() noexcept;


        void execute(nvrhi::DeviceHandle device) noexcept;
        FrameGraph& compile() noexcept;

        explicit FrameGraph(ResourceAllocator& resourceAllocator);

        struct ResourceSlot
        {
            using Version = FrameGraphHandle::Version;
            using Index = int16_t;
            Index rid = 0; // VirtualResource* index in mResources
            Index nid = 0; // ResourceNode* index in mResourceNodes
            Index sid =
                -1; // ResourceNode* index in mResourceNodes for reading subresource's parent
            Version version = 0;
        };


        ResourceSlot& getResourceSlot(FrameGraphHandle handle) noexcept;
        const ResourceSlot& getResourceSlot(FrameGraphHandle handle) const noexcept;
        VirtualResource* getResource(FrameGraphHandle resource_handle) noexcept;


        template<typename Data, typename Setup, typename Execute>
        auto& AddPass(PassType pass_type, const char* name, Setup setup, Execute&& execute)
        {
            // create the FrameGraph pass. Here the data is allocated.
            const auto pass =
                new FrameGraphPassConcrete<Data, Execute>(std::forward<Execute>(execute));

            Builder builder(addPassInternal(name, pass, pass_type));
            setup(builder, const_cast<Data&>(pass->getData()));

            // return a reference to the pass to the user
            return *pass;
        }


        void Reset();


    private:
        DependencyGraph mGraph;
        std::vector<VirtualResource*> mResources;
        std::vector<ResourceSlot> mResourceSlots;
        std::vector<ResourceNode*> mResourceNodes;
        ResourceAllocator& mResourceAllocator;

        template<typename RESOURCE, typename ...Args>
        FrameGraphId<RESOURCE> create(
            const char* name,
            const desc<RESOURCE>& desc,
            Args&&... rest) noexcept
        {
            VirtualResource* vresource = new Resource<RESOURCE, Args...>(
                name,
                desc,
                std::forward<Args>(rest)...);
            return FrameGraphId<RESOURCE>(addResourceInternal(vresource));
        }

        template<typename RESOURCE>
        FrameGraphId<RESOURCE> import(
            const char* name,
            const RESOURCE& resource) noexcept;

        ResourceAllocator& getResourceAllocator() noexcept
        {
            return mResourceAllocator;
        }


    public:
        template<typename RESOURCE>
        auto present(FrameGraphId<RESOURCE> input);

        [[nodiscard]] DependencyGraph& getDependencyGraph()
        {
            return mGraph;
        }

        template<typename RESOURCE>
        FrameGraphId<RESOURCE>
        read(PassNode* passNode, FrameGraphId<RESOURCE> input, Usage usage)
        {
            FrameGraphId<RESOURCE> result(
                readInternal(
                    input,
                    passNode,
                    [this, passNode, usage](ResourceNode* node, VirtualResource* vrsrc)
                    {
                        Resource<RESOURCE>* resource = static_cast<Resource<RESOURCE>*>(vrsrc);
                        return resource->connect(mGraph, node, passNode, usage);
                    }));
            return result;
        }

        template<typename RESOURCE>
        FrameGraphId<RESOURCE>
        write(PassNode* passNode, FrameGraphId<RESOURCE> input, Usage usage)
        {
            FrameGraphId<RESOURCE> result(
                writeInternal(
                    input,
                    passNode,
                    [this, passNode, usage](ResourceNode* node, VirtualResource* vrsrc)
                    {
                        Resource<RESOURCE>* resource = static_cast<Resource<RESOURCE>*>(vrsrc);
                        return resource->connect(mGraph, passNode, node, usage);
                    }));
            return result;
        }

    private:
        std::vector<PassNode*> mPassNodes;
        std::vector<PassNode*>::iterator mActivePassNodesEnd;

        Builder addPassInternal(
            const char* name,
            FrameGraphPassBase* base,
            FrameGraph::PassType type) noexcept;

        void destroyInternal() noexcept;
        void addPresentPass(const std::function<void(Builder&)>& setup) noexcept;


        FrameGraphHandle addResourceInternal(VirtualResource* resource)
        {
            return addSubResourceInternal({}, resource);
        }

        FrameGraphHandle addSubResourceInternal(
            std::vector<FrameGraphHandle> parent,
            VirtualResource* resource) noexcept;

        ResourceNode* getActiveResourceNode(FrameGraphHandle handle) noexcept
        {
            assert(handle);
            const ResourceSlot& slot = getResourceSlot(handle);
            assert((size_t)slot.nid < mResourceNodes.size());
            return mResourceNodes[slot.nid];
        }

        FrameGraphHandle readInternal(
            FrameGraphHandle handle,
            PassNode* passNode,
            const std::function<bool(ResourceNode*, VirtualResource*)>& connect);

        ResourceNode* createNewVersionForSubresourceIfNeeded(
            ResourceNode* node) noexcept;

        FrameGraphHandle createNewVersion(FrameGraphHandle handle) noexcept;

        FrameGraphHandle writeInternal(
            FrameGraphHandle handle,
            PassNode* passNode,
            const std::function<bool(ResourceNode*, VirtualResource*)>& connect);


        /**
         * Marks the current pass as a leaf. Adds a reference to it, so it's not culled.
         * Calling write() on an imported resource automatically adds a side-effect.
         */

        void assertValid(FrameGraphHandle handle) const
        {
            assert(isValid(handle));
            //,"Warning: Resource handle is invalid or uninitialized {id=%u, version=%u}",
            //(int)handle.index, (int)handle.version);
        }

        bool isValid(FrameGraphHandle handle) const
        {
            // Code below is written this way so we can set breakpoints easily.
            if (!handle.isInitialized())
            {
                return false;
            }
            ResourceSlot slot = getResourceSlot(handle);
            if (handle.version != slot.version)
            {
                return false;
            }
            return true;
        }
    };

    template<typename RESOURCE>
    FrameGraphId<RESOURCE> FrameGraph::import(const char* name, const RESOURCE& resource) noexcept
    {
        VirtualResource* vresource(new ImportedResource<RESOURCE>(name, resource));
        return FrameGraphId<RESOURCE>(addResourceInternal(vresource));
    }

    template<typename RESOURCE>
    auto FrameGraph::present(FrameGraphId<RESOURCE> input)
    {
        FrameGraphId<RESOURCE> ret;
        addPresentPass(
            [&](Builder& builder)
            {
                ret = builder.read(input, {});
            });
        return ret;
    }

}
