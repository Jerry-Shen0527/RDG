#pragma once
#include "DescTrait.h"
#include "DependencyGraph.h"
#include "ResourceAllocator.h"
#include "ResourceNode.h"
#include <nvrhi/nvrhi.h>
#include <iostream>

#define NVRHI_ENUM_CLASS_FLAG_OPERATORS(T) \
    inline T operator | (T a, T b) { return T(uint32_t(a) | uint32_t(b)); } \
    inline T operator & (T a, T b) { return T(uint32_t(a) & uint32_t(b)); } /* NOLINT(bugprone-macro-parentheses) */ \
    inline T operator ~ (T a) { return T(~uint32_t(a)); } /* NOLINT(bugprone-macro-parentheses) */ \
    inline bool operator !(T a) { return uint32_t(a) == 0; } \
    inline bool operator ==(T a, uint32_t b) { return uint32_t(a) == b; } \
    inline bool operator !=(T a, uint32_t b) { return uint32_t(a) != b; }

namespace Furnace
{
    class ResourceAllocator;

    enum class Usage:uint32_t
    {
        None = 0,
        DEFAULT_R_USAGE = 1 << 0,
        DEFAULT_W_USAGE = 1 << 1
    };

    NVRHI_ENUM_CLASS_FLAG_OPERATORS(Usage)

#undef NVRHI_ENUM_CLASS_FLAG_OPERATORS


    class VirtualResource
    {
    public:
        // constants
        std::vector<VirtualResource*> parents;
        const char* const name;

        PassNode* first = nullptr; // pass that needs to instantiate the resource
        PassNode* last = nullptr;  // pass that can destroy the resource

        explicit VirtualResource(const char* name) noexcept
            : name(name)
        {
        }


        VirtualResource(const VirtualResource& rhs) noexcept = delete;
        VirtualResource& operator=(const VirtualResource&) = delete;
        virtual ~VirtualResource() noexcept = default;

        // updates first/last/refcount
        void neededByPass(PassNode* pNode) noexcept;

        bool isSubResource() const noexcept
        {
            return !parents.empty();
        }

        /*
         * Called during FrameGraph::compile(), this gives an opportunity for this resource to
         * calculate its effective usage flags.
         */
        virtual void resolveUsage(
            DependencyGraph& graph,
            const ResourceEdge* const* readers,
            size_t count,
            const ResourceEdge* writer) noexcept = 0;

        /* Instantiate the concrete resource */
        virtual void devirtualize(
            ResourceAllocator& resourceAllocator) noexcept = 0;

        /* Destroy the concrete resource */
        virtual void destroy(ResourceAllocator& resourceAllocator) noexcept = 0;

        /* Destroy an Edge instantiated by this resource */
        virtual void destroyEdge(DependencyGraph::Edge* edge) noexcept = 0;

        virtual bool isImported() const noexcept
        {
            return false;
        }

    protected:
        void addOutgoingEdge(ResourceNode* node, ResourceEdge* edge) noexcept;
        void setIncomingEdge(ResourceNode* node, ResourceEdge* edge) noexcept;
        // these exist only so we don't have to include PassNode.h or ResourceNode.h
        static DependencyGraph::Node* toDependencyGraphNode(ResourceNode* node) noexcept;
        static DependencyGraph::Node* toDependencyGraphNode(PassNode* node) noexcept;
        static ResourceEdge* getReaderEdgeForPass(
            ResourceNode* resourceNode,
            PassNode* passNode) noexcept;
        static ResourceEdge* getWriterEdgeForPass(
            ResourceNode* resourceNode,
            PassNode* passNode) noexcept;
    };

    inline void VirtualResource::neededByPass(PassNode* pNode) noexcept
    {
        // figure out which is the first pass to need this resource
        first = first ? first : pNode;
        // figure out which is the last pass to need this resource
        last = pNode;

        // also extend the lifetime of our parent resource if any
        if (!parents.empty())
        {
            for (int i = 0; i < parents.size(); ++i)
            {
                parents[i]->neededByPass(pNode);
            }
        }
    }

    class ResourceEdge : public DependencyGraph::Edge
    {
    public:
        ResourceEdge(
            DependencyGraph& graph,
            DependencyGraph::Node* from,
            DependencyGraph::Node* to,
            Usage usage)
            : Edge(graph, from, to),
              usage(usage)
        {
        }

        Usage usage;
    };

    template<typename RESOURCE, typename...Args>
    class Resource : public VirtualResource
    {
    public:
        RESOURCE resource;
        using Descriptor = desc<RESOURCE>;

        Descriptor descriptor;


        bool detached = false;

        explicit Resource(const char* name, const Descriptor& desc_, Args&&... args_)
            : VirtualResource(name),
              descriptor(desc_),
              args(std::forward<Args>(args_)...)
        {
            setParent_impl(std::forward<Args>(args_)...);
        }

        void resolveUsage(
            DependencyGraph& graph,
            const ResourceEdge* const* edges,
            size_t count,
            const ResourceEdge* writer) noexcept override;

        virtual bool
        connect(DependencyGraph& graph, PassNode* passNode, ResourceNode* resourceNode, Usage u)
        {
            // TODO: we should check that usage flags are correct (e.g. a write flag is not used for
            // reading)
            ResourceEdge* edge =
                getWriterEdgeForPass(resourceNode, passNode);
            if (edge)
            {
                edge->usage = edge->usage | u;
            }
            else
            {
                edge = new ResourceEdge(
                    graph,
                    toDependencyGraphNode(passNode),
                    toDependencyGraphNode(resourceNode),
                    u);
                setIncomingEdge(resourceNode, edge);
            }
            return true;
        }

        // resource Node to pass Node edge (a read from)
        virtual bool
        connect(DependencyGraph& graph, ResourceNode* resourceNode, PassNode* passNode, Usage u)
        {
            // TODO: we should check that usage flags are correct (e.g. a write flag is not used for
            // reading) if passNode is already a reader of resourceNode, then just update the usage
            // flags
            auto edge =
                getReaderEdgeForPass(resourceNode, passNode);
            if (edge)
            {
                edge->usage = edge->usage | u;
            }
            else
            {
                edge = new ResourceEdge(
                    graph,
                    toDependencyGraphNode(resourceNode),
                    toDependencyGraphNode(passNode),
                    u);
                addOutgoingEdge(resourceNode, edge);
            }
            return true;
        }

        void devirtualize(
            ResourceAllocator& resourceAllocator) noexcept override
        {
            auto bare_handles =
                std::apply(
                    [&resourceAllocator](auto&&... extra_args)
                    {
                        return devirtualize_impl(resourceAllocator, std::tuple<>(), extra_args...);
                    },
                    args);
            std::apply(
                [&](auto&&... extra_args)
                {
                    resource = resourceAllocator.create(descriptor, extra_args...);
                },
                bare_handles);
        }

        void destroy(ResourceAllocator& resourceAllocator) noexcept override
        {
            if (detached)
            {
                return;
            }
            resourceAllocator.destroy<RESOURCE>(resource);
        }

        void destroyEdge(DependencyGraph::Edge* edge) noexcept override
        {
            delete static_cast<ResourceEdge*>(edge);
        }

        using ExtraArgs = std::tuple<Args...>;
        ExtraArgs args;

    private:
        template<typename... T1, typename RESC, typename... Arguments>
        static auto devirtualize_impl(
            ResourceAllocator& resourceAllocator,
            std::tuple<T1...>&& tuple,
            RESC* resource,
            Arguments&&... args)
        {
            assert(resource->resource);

            return devirtualize_impl(
                resourceAllocator,
                std::tuple_cat(tuple, std::make_tuple(resource->resource)),
                args...);
        }

        template<typename... T1>
        static auto devirtualize_impl(
            ResourceAllocator& resourceAllocator,
            std::tuple<T1...>&& tuple)
        {
            return std::move(tuple);
        }

        template<typename ... Arguments>
        auto setParent_impl(VirtualResource* resource, Arguments ... args)
        {
            this->parents.push_back(resource);
            setParent_impl(args...);
        }

        void setParent_impl(void)
        {
        }
    };

    template<typename RESOURCE, typename... Args>
    void Resource<RESOURCE, Args...>::resolveUsage(
        DependencyGraph& graph,
        const ResourceEdge* const* edges,
        size_t count,
        const ResourceEdge* writer) noexcept
    {
        // TODO
    }


    /*
         * An imported resource is just like a regular one, except that it's constructed directly from
         * the concrete resource and it, evidently, doesn't create/destroy the concrete resource.
         */
    template<typename RESOURCE>
    class ImportedResource : public Resource<RESOURCE>
    {
    public:
        using Descriptor = desc<RESOURCE>;

        ImportedResource(
            const char* name,
            const RESOURCE& rsrc) noexcept
            : Resource<RESOURCE>(name, Descriptor())
        {
            this->resource = rsrc;
        }

    protected:
        void devirtualize(ResourceAllocator&) noexcept override
        {
            // imported resources don't need to devirtualize
        }

        void destroy(ResourceAllocator&) noexcept override
        {
            // imported resources never destroy the concrete resource
        }

        bool isImported() const noexcept override
        {
            return true;
        }

        bool connect(
            DependencyGraph& graph,
            PassNode* passNode,
            ResourceNode* resourceNode,
            Usage u) override
        {
            return Resource<RESOURCE>::connect(graph, passNode, resourceNode, u);
        }

        bool connect(
            DependencyGraph& graph,
            ResourceNode* resourceNode,
            PassNode* passNode,
            Usage u) override
        {
            return Resource<RESOURCE>::connect(graph, resourceNode, passNode, u);
        }
    };
}
