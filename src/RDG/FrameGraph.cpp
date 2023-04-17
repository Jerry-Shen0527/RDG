#include<RDG/RDG/FrameGraph.h>

#include <algorithm>
#include <iostream>

#include "RDG/RDG/DependencyGraph.h"
#include "RDG/RDG/FrameGraphResource.h"
#include "PassNode.h"

#include <RDG/RDG/FrameGraphPass.h>

namespace Furnace
{
    FrameGraphPassBase::~FrameGraphPassBase() noexcept
    {
#ifdef _DEBUG
        std::cout << "Pass killed" << std::endl;
#endif
    }

    inline FrameGraph::Builder::Builder(FrameGraph& fg, PassNode* node) noexcept
        : mFrameGraph(fg),
          mPassNode(node)
    {
    }

    FrameGraph::Builder FrameGraph::addPassInternal(
        const char* name,
        FrameGraphPassBase* base,
        PassType type) noexcept
    {
        // record in our pass list and create the builder
        PassNode* node;

        switch (type)
        {
            case PassType::Render: node = new RenderPassNode(*this, name, base);
                break;
            case PassType::Compute: node = new ComputePassNode(*this, name, base);
                break;
            case PassType::RayTracing: node = new RayTracingPassNode(*this, name, base);
                break;
#ifdef RDG_WITH_CUDA
            case PassType::Cuda: node = new CudaPassNode(*this, name, base); break;
#endif
#ifdef RDG_WITH_OPTIX
            case PassType::OptiX: node = new OptiXPassNode(*this, name, base); break;
#endif
            default: ;
        }
        base->setNode(node);
        mPassNodes.push_back(node);
        return { *this, node };
    }

    void FrameGraph::destroyInternal() noexcept
    {
        for (PassNode* mPassNode : mPassNodes)
        {
            delete mPassNode;
        }

        for (ResourceNode* resourceNode : mResourceNodes)
        {
            delete resourceNode;
        }

        for (VirtualResource* resourceNode : mResources)
        {
            delete resourceNode;
        }
    }

    void FrameGraph::addPresentPass(const std::function<void(Builder&)>& setup) noexcept
    {
        auto node = new PresentPassNode(*this);
        mPassNodes.push_back(node);
        Builder builder(*this, node);
        setup(builder);
        builder.sideEffect();
    }

    FrameGraphHandle FrameGraph::addSubResourceInternal(
        std::vector<FrameGraphHandle> parent,
        VirtualResource* resource) noexcept
    {
        FrameGraphHandle handle(mResourceSlots.size());
        ResourceSlot& slot = mResourceSlots.emplace_back();
        slot.rid = mResources.size();
        slot.nid = mResourceNodes.size();
        mResources.push_back(resource);
        auto pNode = new ResourceNode(*this, handle, parent);
        mResourceNodes.push_back(pNode);
        return handle;
    }

    FrameGraphHandle FrameGraph::readInternal(
        FrameGraphHandle handle,
        PassNode* passNode,
        const std::function<bool(ResourceNode*, VirtualResource*)>& connect)
    {
        assertValid(handle);

        VirtualResource* const resource = getResource(handle);
        ResourceNode* const node = getActiveResourceNode(handle);

        // Check preconditions
        bool passAlreadyAWriter = node->hasWriteFrom(passNode);
        assert(!passAlreadyAWriter);
        //"Pass \"%s\" already writes to \"%s\"",
        //passNode->getName(),
        //node->getName());

        //if (!node->hasWriterPass() && !resource->isImported())
        //{
        //    // TODO: we're attempting to read from a resource that was never written and is not
        //    //       imported either, so it can't have valid data in it.
        //    //       Should this be an error?
        //}

        // Allow reading from empty resources. But do nothing. Don't connect.
        if (connect(node, resource))
        {
            if (resource->isSubResource())
            {
                // this is a read() from a subresource, so we need to add a "read" from the
                // parent's node to the subresource -- but we may have two parent nodes, one for
                // reads and one for writes, so we need to use the one for reads.

                auto parentNodes = node->getParentNodes();

                for (int i = 0; i < parentNodes.size(); ++i)
                {
                    auto* parentNode = parentNodes[i];
                    const ResourceSlot& slot = getResourceSlot(parentNode->resourceHandle);
                    if (slot.sid >= 0)
                    {
                        // we have a parent's node for reads, use that one
                        parentNode = mResourceNodes[slot.sid];
                    }
                    node->setParentReadDependency(parentNode);
                }
            }
            else
            {
                // we're reading from a top-level resource (i.e. not a subresource), but this
                // resource is a parent of some subresource, and it might exist as a version for
                // writing, in this case we need to add a dependency from its "read" version to
                // itself.
                ResourceSlot& slot = getResourceSlot(handle);
                if (slot.sid >= 0)
                {
                    node->setParentReadDependency(mResourceNodes[slot.sid]);
                }
            }

            // if a resource has a subresource, then its handle becomes valid again as soon as
            // it's used.
            ResourceSlot& slot = getResourceSlot(handle);
            if (slot.sid >= 0)
            {
                // we can now forget the "read" parent node, which becomes the current one again
                // until the next write.
                slot.sid = -1;
            }

            return handle;
        }

        return {};
    }

    ResourceNode* FrameGraph::createNewVersionForSubresourceIfNeeded(ResourceNode* node) noexcept
    {
        ResourceSlot& slot = getResourceSlot(node->resourceHandle);
        if (slot.sid < 0)
        {
            // if we don't already have a new ResourceNode for this resource, create one.
            // we keep the old ResourceNode index, so we can direct all the reads to it.
            slot.sid = slot.nid;              // record the current ResourceNode of the parent
            slot.nid = mResourceNodes.size(); // create the new parent node
            node =
                new ResourceNode(*this, node->resourceHandle, node->getParentHandles());
            mResourceNodes.push_back(node);
        }
        return node;
    }

    FrameGraphHandle FrameGraph::createNewVersion(FrameGraphHandle handle) noexcept
    {
        assert(handle);
        ResourceNode* const node = getActiveResourceNode(handle);
        assert(node);
        auto parent = node->getParentHandles();
        ResourceSlot& slot = getResourceSlot(handle);
        slot.version = ++handle.version;  // increase the parent's version
        slot.nid = mResourceNodes.size(); // create the new parent node
        auto newNode = new ResourceNode(*this, handle, parent);
        mResourceNodes.push_back(newNode);
        return handle;
    }

    FrameGraphHandle FrameGraph::writeInternal(
        FrameGraphHandle handle,
        PassNode* passNode,
        const std::function<bool(ResourceNode*, VirtualResource*)>& connect)
    {
        assertValid(handle);

        VirtualResource* const resource = getResource(handle);
        ResourceNode* node = getActiveResourceNode(handle);
        std::vector<ResourceNode*> parentNodes = node->getParentNodes();

        // if we're writing into a subresource, we also need to add a "write" from the
        // subresource node to a new version of the parent's node, if we don't already have one.
        if (resource->isSubResource())
        {
            assert(!parentNodes.empty());
            // this could be a subresource from a subresource, and in this case, we want the
            // oldest ancestor, that is, the node that started it all.
            parentNodes = ResourceNode::getAncestorNode(parentNodes);
            // FIXME: do we need the equivalent of hasWriterPass() test below

            for (int i = 0; i < parentNodes.size(); ++i)
            {
                parentNodes[i] = createNewVersionForSubresourceIfNeeded(parentNodes[i]);
            }
        }

        // if this node already writes to this resource, just update the used bits
        if (!node->hasWriteFrom(passNode))
        {
            if (!node->hasWriterPass() && !node->hasReaders())
            {
                // FIXME: should this also take subresource writes into account
                // if we don't already have a writer or a reader, it just means the resource was
                // just created and was never written to, so we don't need a new node or
                // increase the version number
            }
            else
            {
                handle = createNewVersion(handle);
                // refresh the node
                node = getActiveResourceNode(handle);
            }
        }

        if (connect(node, resource))
        {
            if (resource->isSubResource())
            {
                for (int i = 0; i < parentNodes.size(); ++i)
                {
                    node->setParentWriteDependency(parentNodes[i]);
                }
            }
            if (resource->isImported())
            {
                // writing to an imported resource implies a side-effect
                passNode->makeTarget();
            }
            return handle;
        }
        else
        {
            // FIXME: we need to undo everything we did to this point
        }

        return {};
    }


    void FrameGraph::Builder::sideEffect() noexcept
    {
        mPassNode->makeTarget();
    }

    uint32_t FrameGraph::Builder::declareRenderPass(
        const char* name,
        const FrameGraphRenderPass::Descriptor& desc)
    {
        return static_cast<RenderPassNode*>(mPassNode)->declareRenderTarget(
            mFrameGraph,
            *this,
            name,
            desc);
    }

    void FrameGraph::Builder::declareComputePass(
        const char* name,
        const FrameGraphComputePass::Descriptor& desc)
    {
        static_cast<ComputePassNode*>(mPassNode)->declareComputePass(
            mFrameGraph,
            *this,
            name,
            desc);
    }

    void FrameGraph::Builder::declareRayTracingPass(
        const char* name,
        const FrameGraphRayTracingPass::Descriptor& desc)
    {
        static_cast<RayTracingPassNode*>(mPassNode)->declareRayTracingPass(
            mFrameGraph,
            *this,
            name,
            desc);
    }
#ifdef RDG_WITH_CUDA
    void FrameGraph::Builder::declareCudaPass(
        const char* name,
        const FrameGraphCudaPass::Descriptor& desc)
    {
        static_cast<CudaPassNode*>(mPassNode)->declareCudaPass(mFrameGraph, *this, name, desc);
    }
#endif

#ifdef RDG_WITH_OPTIX
    void FrameGraph::Builder::declareOptiXPass(
        const char* name,
        const FrameGraphOptiXPass::Descriptor& desc)
    {
        static_cast<OptiXPassNode*>(mPassNode)->declareOptiXPass(mFrameGraph, *this, name, desc);
    }
#endif

    void FrameGraph::execute(nvrhi::DeviceHandle device) noexcept
    {
        const auto& passNodes = mPassNodes;
        //auto& resourceAllocator = mResourceAllocator;

        auto first = passNodes.begin();
        const auto activePassNodesEnd = mActivePassNodesEnd;
        while (first != activePassNodesEnd)
        {
            PassNode* const node = *first;
            first++;
            //assert(!node->isCulled());

            //driver.pushGroupMarker(node->getName());

            //// devirtualize resourcesList
            for (VirtualResource* resource : node->devirtualize)
            {
                assert(resource->first == node);
                resource->devirtualize(mResourceAllocator);
            }
            node->devirtualize.clear();

            // call execute
            FrameGraphResources resources(*this, *node);
            node->execute(resources, device);

            // destroy concrete resources
            for (VirtualResource* resource : node->destroy)
            {
                assert(resource->last == node);
                resource->destroy(mResourceAllocator);
            }
            node->destroy.clear();
        }

        device->waitForIdle();
    }

    FrameGraph& FrameGraph::compile() noexcept
    {
        DependencyGraph& dependencyGraph = mGraph;

        // first we cull unreachable nodes
        dependencyGraph.cull();

        mActivePassNodesEnd = std::stable_partition(
            mPassNodes.begin(),
            mPassNodes.end(),
            [](const auto& pPassNode)
            {
                return !pPassNode->isCulled();
            });

        auto first = mPassNodes.begin();
        const auto activePassNodesEnd = mActivePassNodesEnd;
        while (first != activePassNodesEnd)
        {
            PassNode* const passNode = *first;
            first++;
            assert(!passNode->isCulled());

            const auto& reads = dependencyGraph.getIncomingEdges(passNode);
            for (const auto& edge : reads)
            {
                // all incoming edges should be valid by construction
                assert(dependencyGraph.isEdgeValid(edge));
                auto pNode = static_cast<ResourceNode*>(dependencyGraph.getNode(edge->from));
                passNode->registerResource(pNode->resourceHandle);
            }

            const auto& writes = dependencyGraph.getOutgoingEdges(passNode);
            for (const auto& edge : writes)
            {
                // an outgoing edge might be invalid if the node it points to has been culled
                // but, because we are not culled and we're a pass, we add a reference to
                // the resource we are writing to.
                auto pNode = static_cast<ResourceNode*>(dependencyGraph.getNode(edge->to));
                passNode->registerResource(pNode->resourceHandle);
            }

            passNode->resolve();
        }

        // add resource to de-virtualize or destroy to the corresponding list for each active pass
        for (auto* pResource : mResources)
        {
            VirtualResource* resource = pResource;
            {
                PassNode* pFirst = resource->first;
                PassNode* pLast = resource->last;
                assert(!pFirst == !pLast);
                if (pFirst && pLast)
                {
                    assert(!pFirst->isCulled());
                    assert(!pLast->isCulled());
                    pFirst->devirtualize.push_back(resource);
                    pLast->destroy.push_back(resource);
                }
            }
        }

        /*
         * Resolve Usage bits
         */
        for (auto& pNode : mResourceNodes)
        {
            // we can't use isCulled() here because some culled resource are still active
            // we could use "getResource(pNode->resourceHandle)->refcount" but that's expensive.
            // We also can't remove or reorder this array, as handles are indices to it.
            // We might need to build an array of indices to active resources.
            pNode->resolveResourceUsage(dependencyGraph);
        }

        return *this;
    }

    FrameGraph::FrameGraph(ResourceAllocator& resourceAllocator)
        : mResourceAllocator(resourceAllocator)
    {
        mResourceSlots.reserve(256);
        mResources.reserve(256);
        mResourceNodes.reserve(256);
        mPassNodes.reserve(64);
    }

    FrameGraph::~FrameGraph() noexcept
    {
        destroyInternal();
#ifdef _DEBUG
        std::cout << "Graph killed" << std::endl;
#endif
    }

    //const FrameGraph::ResourceSlot& FrameGraph::getResourceSlot(
    //    FrameGraphHandle handle) const noexcept
    //{
    //    assert((size_t)handle.index < mResourceSlots.size());
    //    assert((size_t)mResourceSlots[handle.index].rid < mResources.size());
    //    assert((size_t)mResourceSlots[handle.index].nid < mResourceNodes.size());
    //    return mResourceSlots[handle.index];
    //}


    FrameGraph::ResourceSlot& FrameGraph::getResourceSlot(FrameGraphHandle handle) noexcept
    {
        assert((size_t)handle.index < mResourceSlots.size());
        assert((size_t)mResourceSlots[handle.index].rid < mResources.size());
        assert((size_t)mResourceSlots[handle.index].nid < mResourceNodes.size());
        return mResourceSlots[handle.index];
    }

    const FrameGraph::ResourceSlot& FrameGraph::getResourceSlot(
        FrameGraphHandle handle) const noexcept
    {
        return const_cast<FrameGraph*>(this)->getResourceSlot(handle);
    }

    /**
     * \brief This is a tour around. First use the handle to find the resouce slot, then use the slot id to find the resource.
     * \param handle 
     * \return 
     */
    VirtualResource* FrameGraph::getResource(FrameGraphHandle handle) noexcept
    {
        assert(handle.isInitialized());
        const ResourceSlot& slot = getResourceSlot(handle);
        assert((size_t)slot.rid < mResources.size());
        return mResources[slot.rid];
    }

    void FrameGraph::Reset()
    {
        destroyInternal();
        mPassNodes.clear();
        mResourceNodes.clear();
        mResources.clear();
        mResourceSlots.clear();

        mGraph.clear();
    }
}
