#include<RDG/RDG/DependencyGraph.h>

#include <algorithm>
#include <iterator>

#include "PassNode.h"

namespace Furnace
{
    bool DependencyGraph::isEdgeValid(Edge* edge)
    {
        auto& nodes = mNodes;
        const Node* from = nodes[edge->from];
        const Node* to = nodes[edge->to];
        return !from->isCulled() && !to->isCulled();
    }

    DependencyGraph::EdgeContainer DependencyGraph::getOutgoingEdges(PassNode* node) const
    {
        auto result = EdgeContainer();
        const NodeID nodeId = node->getId();
        std::copy_if(
            mEdges.begin(),
            mEdges.end(),
            std::back_insert_iterator(result),
            [nodeId](auto edge)
            {
                return edge->from == nodeId;
            });
        return result;
    }

    void DependencyGraph::cull() noexcept
    {
        auto& nodes = mNodes;
        auto& edges = mEdges;

        // update reference counts
        for (Edge* const pEdge : edges)
        {
            Node* node = nodes[pEdge->from];
            node->mRefCount++;
        }

        // cull nodes with a 0 reference count
        auto stack = NodeContainer();
        for (Node* const pNode : nodes)
        {
            if (pNode->getRefCount() == 0)
            {
                stack.push_back(pNode);
            }
        }
        while (!stack.empty())
        {
            Node* const pNode = stack.back();
            stack.pop_back();
            const EdgeContainer& incoming = getIncomingEdges(pNode);
            for (Edge* edge : incoming)
            {
                Node* pLinkedNode = getNode(edge->from);
                if (--pLinkedNode->mRefCount == 0)
                {
                    stack.push_back(pLinkedNode);
                }
            }
        }
    }

    char const* DependencyGraph::Node::getName() const noexcept
    {
        return "Unknown";
    }

    void DependencyGraph::Node::makeTarget() noexcept
    {
        assert(mRefCount == 0 || mRefCount == TARGET);
        mRefCount = TARGET;
    }

    void DependencyGraph::link(Edge* edge)
    {
        auto& edges = mEdges;
        edges.push_back(edge);
    }

    void DependencyGraph::registerNode(Node* node, NodeID id)
    {
        // Node* is not fully constructed here
        assert(id == mNodes.size());

        // here we manually grow the fixed-size vector
        auto& nodes = mNodes;
        nodes.push_back(node);
    }

    DependencyGraph::EdgeContainer DependencyGraph::getIncomingEdges(const Node* node) const
    {
        auto result = EdgeContainer();
        const NodeID nodeId = node->getId();
        std::copy_if(
            mEdges.begin(),
            mEdges.end(),
            std::back_insert_iterator(result),
            [nodeId](auto edge)
            {
                return edge->to == nodeId;
            });
        return result;
    }

    const DependencyGraph::Node* DependencyGraph::getNode(NodeID id) const noexcept
    {
        return mNodes[id];
    }

    DependencyGraph::Node* DependencyGraph::getNode(NodeID id) noexcept
    {
        return mNodes[id];
    }

    void DependencyGraph::clear() noexcept
    {
        mEdges.clear();
        mNodes.clear();
    }
}
