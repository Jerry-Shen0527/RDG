/*
 * Copyright (C) 2021 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <vector>
#include <cstdint>

namespace Furnace
{
    class PassNode;

    class DependencyGraph
    {
    public:
        struct Node;
        struct Edge;

        using EdgeContainer = std::vector<Edge*>;
        using NodeContainer = std::vector<Node*>;


        DependencyGraph() noexcept
        {
        }

        ~DependencyGraph() noexcept
        {
        }

        bool isEdgeValid(Edge* edge);
        EdgeContainer getOutgoingEdges(PassNode* pass_node) const;
        void cull() noexcept;


        DependencyGraph(const DependencyGraph&) noexcept = delete;
        DependencyGraph& operator=(const DependencyGraph&) noexcept = delete;


        using NodeID = uint32_t;

        uint32_t generateNodeId()
        {
            return mNodes.size();
        }


        /**
         * A generic node
         */
        struct Node
        {
            friend class DependencyGraph;
        public:
            explicit Node(DependencyGraph& graph) noexcept
                : mId(graph.generateNodeId())
            {
                graph.registerNode(this, mId);
            }

            /**
             * Creates a Node and adds it to the graph. The caller keeps ownership of the Node
             * object, which is only safe to destroy after calling DependencyGraph::clear().
             * @param graph DependencyGraph pointer to add the Node to.
             */
            //explicit Node(DependencyGraph& graph) noexcept;

            // Nodes can't be copied
            Node(const Node&) noexcept = delete;
            Node& operator=(const Node&) noexcept = delete;

            NodeID getId() const
            {
                return this->mId;
            }

            uint32_t getRefCount() noexcept
            {
                return mRefCount;
            }

            //! Nodes can be moved
            Node(Node&&) noexcept = default;

            virtual ~Node() noexcept = default;

            bool isCulled() const noexcept
            {
                return mRefCount == 0;
            }

            virtual const char* getName() const noexcept;
            void makeTarget() noexcept;

        private:
            static const constexpr uint32_t TARGET = 0x80000000u;

            uint32_t mRefCount = 0; // how many references to us
            const NodeID mId;       // unique id
        };

        void link(Edge* edge);

        struct Edge
        {
            // An Edge can't be modified after it's created (e.g. by copying into it)
            const NodeID from;
            const NodeID to;

            Edge(DependencyGraph& graph, Node* from, Node* to)
                : from(from->getId()),
                  to(to->getId())
            {
                graph.link(this);
            }

            // Edge can't be copied or moved, this is to allow subclassing safely.
            // Subclasses can hold their own data.
            Edge(const Edge& rhs) noexcept = delete;
            Edge& operator=(const Edge& rhs) noexcept = delete;
        };


        void registerNode(Node* node, NodeID id);

        EdgeContainer getIncomingEdges(const Node* node)const;

        const Node* getNode(NodeID id) const noexcept;

        Node* getNode(NodeID id) noexcept;

    void clear() noexcept;

    private:
        NodeContainer mNodes;
        EdgeContainer mEdges;
    };
}
