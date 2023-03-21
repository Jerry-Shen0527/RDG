#pragma once

#include <set>

#include "RDG/RDG/DependencyGraph.h"
#include "FrameGraphId.h"

namespace Furnace
{
    class ResourceEdge;

    class ResourceNode : public DependencyGraph::Node
    {
    public:
        ResourceNode(
            FrameGraph& fg,
            FrameGraphHandle h,
            std::vector<FrameGraphHandle> ParentHandles) noexcept;
        ~ResourceNode() noexcept override;
        std::vector<ResourceNode*> getParentNodes() noexcept;
        std::vector<FrameGraphHandle> getParentHandles() const noexcept;

        ResourceNode(const ResourceNode&) = delete;
        ResourceNode& operator=(const ResourceNode&) = delete;

        FrameGraphHandle resourceHandle;
        void resolveResourceUsage(DependencyGraph& graph);

        ResourceEdge* getWriterEdgeForPass(const PassNode* node) const noexcept;
        ResourceEdge* getReaderEdgeForPass(PassNode* pass_node) const noexcept;


        void setParentReadDependency(ResourceNode* parent) noexcept;


        bool hasWriteFrom(const PassNode* node) const noexcept
        {
            return bool(getWriterEdgeForPass(node));
        }

        // is at least one PassNode reading from this ResourceNode
        bool hasReaders() const noexcept
        {
            return !mReaderPasses.empty();
        }

        void setParentWriteDependency(ResourceNode* parent) noexcept;

        void setParents(const std::vector<FrameGraphHandle>& ParentHandle) noexcept
        {
            mParentHandles = ParentHandle;
        }

        // is a PassNode writing to this ResourceNode
        bool hasWriterPass() const noexcept
        {
            return mWriterPass != nullptr;
        }

        static FrameGraphHandle getHandle(const ResourceNode* node) noexcept
        {
            return node ? node->resourceHandle : FrameGraphHandle{};
        }

        void addOutgoingEdge(ResourceEdge* edge) noexcept;

        void setIncomingEdge(ResourceEdge* edge) noexcept;

        static std::vector<ResourceNode*> getAncestorNode(std::vector<ResourceNode*> node) noexcept
        {

            if (node.empty())
            {
                return node;
            }
            std::set<ResourceNode*> ret(node.begin(), node.end());

            for (int i = 0; i < node.size(); ++i)
            {
                auto ancestor = getAncestorNode(node[i]->getParentNodes());
                for (int j = 0; j < ancestor.size(); ++j)
                {
                    ret.insert(ancestor[j]);
                }
            }
            return std::vector<ResourceNode*>(ret.begin(), ret.end());
        }

    private:
        FrameGraph& mFrameGraph;
        std::vector<FrameGraphHandle> mParentHandles;
        DependencyGraph::Edge* mParentReadEdge = nullptr;
        DependencyGraph::Edge* mParentWriteEdge = nullptr;
        DependencyGraph::Edge* mForwardedEdge = nullptr;

        //Using this, the resource node can 'look up itself' by getActiveResourceNode(handle)

        // Can be read by a lot of passes, while only writable by one pass?
        std::vector<ResourceEdge*> mReaderPasses;
        ResourceEdge* mWriterPass = nullptr;
    };
}
