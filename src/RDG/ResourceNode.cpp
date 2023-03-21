#include "PassNode.h"
#include "RDG/RDG/FrameGraph.h"

namespace Furnace
{
    ResourceNode::ResourceNode(
        FrameGraph& fg,
        FrameGraphHandle h,
        std::vector<FrameGraphHandle> ParentHandles) noexcept
        : Node(fg.getDependencyGraph()),
          resourceHandle(h),
          mFrameGraph(fg),
          mParentHandles(ParentHandles)
    {
    }

    ResourceNode::~ResourceNode() noexcept
    {
        VirtualResource* resource = mFrameGraph.getResource(resourceHandle);
        assert(resource);
        resource->destroyEdge(mWriterPass);
        for (auto* pEdge : mReaderPasses)
        {
            resource->destroyEdge(pEdge);
        }
        delete mParentReadEdge;
        delete mParentWriteEdge;
        delete mForwardedEdge;
    }

    std::vector<ResourceNode*> ResourceNode::getParentNodes() noexcept
    {
        std::vector<ResourceNode*> ret;
        if (!mParentHandles.empty())
        {
            for (int i = 0; i < mParentHandles.size(); ++i)
            {
                ResourceNode* const parentNode =
                    mFrameGraph.getActiveResourceNode(mParentHandles[i]);
                ret.push_back(parentNode);
                assert(mParentHandles[i] == ResourceNode::getHandle(parentNode));
            }
        }
        return ret;
    }


    std::vector<FrameGraphHandle> ResourceNode::getParentHandles() const noexcept
    {
        return mParentHandles;
    }

    void ResourceNode::resolveResourceUsage(DependencyGraph& graph)
    {
        VirtualResource* pResource = mFrameGraph.getResource(resourceHandle);
        assert(pResource);
        pResource->resolveUsage(graph, mReaderPasses.data(), mReaderPasses.size(), mWriterPass);
    }

    ResourceEdge* ResourceNode::getWriterEdgeForPass(const PassNode* node) const noexcept
    {
        return mWriterPass && mWriterPass->from == node->getId() ? mWriterPass : nullptr;
    }

    ResourceEdge* ResourceNode::getReaderEdgeForPass(PassNode* node) const noexcept
    {
        auto pos = std::find_if(
            mReaderPasses.begin(),
            mReaderPasses.end(),
            [node](const ResourceEdge* edge)
            {
                return edge->to == node->getId();
            });
        return pos != mReaderPasses.end() ? *pos : nullptr;
    }

    void ResourceNode::setParentReadDependency(ResourceNode* parent) noexcept
    {
        if (!mParentReadEdge)
        {
            mParentReadEdge = new DependencyGraph::Edge(
                mFrameGraph.getDependencyGraph(),
                parent,
                this);
        }
    }

    void ResourceNode::setParentWriteDependency(ResourceNode* parent) noexcept
    {
        if (!mParentWriteEdge)
        {
            mParentWriteEdge = new DependencyGraph::Edge(
                mFrameGraph.getDependencyGraph(),
                this,
                parent);
        }
    }

    void ResourceNode::addOutgoingEdge(ResourceEdge* edge) noexcept
    {
        mReaderPasses.push_back(edge);
    }

    void ResourceNode::setIncomingEdge(ResourceEdge* edge) noexcept
    {
        assert(mWriterPass == nullptr);
        mWriterPass = edge;
    }
} // namespace Furnace
