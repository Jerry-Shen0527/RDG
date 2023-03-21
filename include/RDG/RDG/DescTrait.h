#pragma once

#include <nvrhi/nvrhi.h>
#include "macro_map.h"

namespace nvrhi
{
    using CommandListDesc = CommandListParameters;

    struct BindlessLayoutHandle : public BindingLayoutHandle
    {
        using BindingLayoutHandle::BindingLayoutHandle;

        BindlessLayoutHandle(BindingLayoutHandle&& handle)
            : BindingLayoutHandle(std::move(handle))
        {
        }

        BindlessLayoutHandle(const BindingLayoutHandle& handle)
            : BindingLayoutHandle(handle)
        {
        }

        BindingLayoutHandle operator()() const
        {
            return *this;
        }

        BindingLayoutHandle& operator()()
        {
            return *this;
        }
    };

    struct DescriptorTableDesc
    {
        friend bool operator==(const DescriptorTableDesc& lhs, const DescriptorTableDesc& rhs)
        {
            return false;
        }

        friend bool operator!=(const DescriptorTableDesc& lhs, const DescriptorTableDesc& rhs)
        {
            return !(lhs == rhs);
        }
    };

    using IBindlessLayout = nvrhi::IBindingLayout;

    namespace rt
    {
        using ShaderTableDesc = PipelineHandle;
    }
}

namespace Furnace
{
#define NVRHI_RESOURCE_LIST                                                                     \
    Texture, GraphicsPipeline, Framebuffer, CommandList, BindingLayout, BindlessLayout, Buffer, \
        BindingSet, ComputePipeline, Sampler, DescriptorTable

#define NVRHI_RT_RESOURCE_LIST Pipeline, ShaderTable, AccelStruct
#define NVRHI_NAMESPACE_WRAP(RSC) WRAPPED_MACRO(nvrhi, RSC)
#define NVRHI_RT_NAMESPACE_WRAP(RSC) WRAPPED_MACRO(nvrhi::rt, RSC)

#define RESOURCE_LIST NVRHI_RESOURCE_LIST

#define MAP_TO_NAMESPACE(NSP) DRJIT_MAP(NSP##_NAMESPACE_WRAP, NSP##_RESOURCE_LIST);
#define MAP_TO_ALL_NAMESPACE  MAP_TO_NAMESPACE(NVRHI) MAP_TO_NAMESPACE(NVRHI_RT) 

#define DESC_HANDLE_TRAIT(NAMESPACE, RESOURCE)      \
    template<>                                      \
    struct ResouceDesc<NAMESPACE::RESOURCE##Handle> \
    {                                               \
        using Desc = NAMESPACE::RESOURCE##Desc;     \
    };

#define HANDLE_DESC_TRAIT(NAMESPACE, RESOURCE)      \
    template<>                                      \
    struct DescResouce<NAMESPACE::RESOURCE##Desc> \
    {                                               \
        using Resource = NAMESPACE::RESOURCE##Handle;     \
    };


    template<typename RESOURCE>
    struct ResouceDesc
    {
        using Desc = void;
    };

    template<typename DESC>
    struct DescResouce
    {
        using Resource = void;
    };

#define WRAPPED_MACRO DESC_HANDLE_TRAIT
    MAP_TO_ALL_NAMESPACE

#define WRAPPED_MACRO HANDLE_DESC_TRAIT
    MAP_TO_ALL_NAMESPACE

    template<typename RESOURCE>
    using desc = typename ResouceDesc<RESOURCE>::Desc;

    template<typename DESC>
    using resc = typename DescResouce<DESC>::Resource;
}
