#include <RDG/RDG/Resource.h>

#include "PassNode.h"

namespace Furnace
{
    void VirtualResource::addOutgoingEdge(ResourceNode* node, ResourceEdge* edge) noexcept
    {
        node->addOutgoingEdge(edge);
    }

    void VirtualResource::setIncomingEdge(ResourceNode* node, ResourceEdge* edge) noexcept
    {
        node->setIncomingEdge(edge);
    }

    DependencyGraph::Node* VirtualResource::toDependencyGraphNode(ResourceNode* node) noexcept
    {
        return node;
    }

    DependencyGraph::Node* VirtualResource::toDependencyGraphNode(PassNode* node) noexcept
    {
        return node;
    }

    ResourceEdge* VirtualResource::getReaderEdgeForPass(
        ResourceNode* resourceNode,
        PassNode* passNode) noexcept
    {
        return resourceNode->getReaderEdgeForPass(passNode);
    }

    ResourceEdge* VirtualResource::getWriterEdgeForPass(
        ResourceNode* resourceNode,
        PassNode* passNode) noexcept
    {
        return resourceNode->getWriterEdgeForPass(passNode);
    }
}
