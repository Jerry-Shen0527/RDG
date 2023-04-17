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

#include <nvrhi/nvrhi.h>

#include "ResourceAllocator.h"
#include "donut/engine/View.h"
#include "RDG/OptiXSupport.h"

#define NVRHI_ENUM_CLASS_FLAG_OPERATORS(T) \
    inline T operator | (T a, T b) { return T(uint32_t(a) | uint32_t(b)); } \
    inline T operator & (T a, T b) { return T(uint32_t(a) & uint32_t(b)); } /* NOLINT(bugprone-macro-parentheses) */ \
    inline T operator ~ (T a) { return T(~uint32_t(a)); } /* NOLINT(bugprone-macro-parentheses) */ \
    inline bool operator !(T a) { return uint32_t(a) == 0; } \
    inline bool operator ==(T a, uint32_t b) { return uint32_t(a) == b; } \
    inline bool operator !=(T a, uint32_t b) { return uint32_t(a) != b; }

namespace Furnace
{
    enum class TargetBufferFlags : uint32_t
    {
        NONE = 0x0u,
        //!< No buffer selected.
        COLOR0 = 0x00000001u,
        //!< Color buffer selected.
        COLOR1 = 0x00000002u,
        //!< Color buffer selected.
        COLOR2 = 0x00000004u,
        //!< Color buffer selected.
        COLOR3 = 0x00000008u,
        //!< Color buffer selected.
        COLOR4 = 0x00000010u,
        //!< Color buffer selected.
        COLOR5 = 0x00000020u,
        //!< Color buffer selected.
        COLOR6 = 0x00000040u,
        //!< Color buffer selected.
        COLOR7 = 0x00000080u,
        //!< Color buffer selected.

        COLOR = COLOR0,
        //!< \deprecated
        COLOR_ALL = COLOR0 | COLOR1 | COLOR2 | COLOR3 | COLOR4 | COLOR5 | COLOR6 | COLOR7,
        DEPTH = 0x10000000u,
        //!< Depth buffer selected.
        STENCIL = 0x20000000u,
        //!< Stencil buffer selected.
        DEPTH_AND_STENCIL = DEPTH | STENCIL,
        //!< depth and stencil buffer selected.
        ALL = COLOR_ALL | DEPTH | STENCIL //!< Color, depth and stencil buffer selected.
    };

    NVRHI_ENUM_CLASS_FLAG_OPERATORS(TargetBufferFlags)

    /**
     * FrameGraphRenderPass is used to draw into a set of nvrhi::TextureHandle resources.
     * These are transient objects that exist inside a pass only.
     */
    struct FrameGraphRenderPass
    {
        static constexpr size_t ATTACHMENT_COUNT = nvrhi::c_MaxRenderTargets + 2;

        struct Attachments
        {
            union
            {
                FrameGraphId<nvrhi::TextureHandle> array[ATTACHMENT_COUNT] = {};

                struct
                {
                    FrameGraphId<nvrhi::TextureHandle> color[nvrhi::c_MaxRenderTargets];
                    FrameGraphId<nvrhi::TextureHandle> depth;
                    FrameGraphId<nvrhi::TextureHandle> stencil;
                };
            };
        };

        struct Descriptor
        {
            Descriptor()
            {
                binding_layouts = BindingLayoutVector(nvrhi::c_MaxBindingLayouts);
            }

            Attachments attachments{};
            std::shared_ptr<donut::engine::IView> viewport;
            nvrhi::Color clearColor{};
            float clearDepth{};
            uint8_t samples = 0; // # of samples (0 = unset, default)
            TargetBufferFlags clearFlags = TargetBufferFlags::ALL;

            nvrhi::GraphicsPipelineDesc pipelineDesc;


            Descriptor& addBindingLayout(const nvrhi::BindingLayoutHandle& handle, int idx)
            {
                binding_layouts[idx] = handle;
                return *this;
            }

            Descriptor& addBindlessLayout(const nvrhi::BindlessLayoutHandle& handle, int idx)
            {
                binding_layouts[idx] = handle;
                return *this;
            }


            friend class RenderPassNode;

        private:
            void resolve(ResourceAllocator& resourceAllocator)
            {
                pipelineDesc.bindingLayouts = binding_layouts;

                for (int i = nvrhi::c_MaxBindingLayouts - 1; i >= 0; --i)
                {
                    if (!pipelineDesc.bindingLayouts[i])
                    {
                        pipelineDesc.bindingLayouts.resize(i);
                    }
                }
            }


            using BindingLayoutVector = nvrhi::static_vector<
                nvrhi::BindingLayoutHandle, nvrhi::c_MaxBindingLayouts>;
            BindingLayoutVector binding_layouts;
        };
    };

    struct FrameGraphComputePass
    {
        struct Descriptor
        {
            Descriptor()
            {
                binding_layouts = BindingLayoutVector(nvrhi::c_MaxBindingLayouts);
            }

            nvrhi::ComputePipelineDesc pipelineDesc;

            Descriptor& addBindingLayout(const nvrhi::BindingLayoutHandle& handle, int idx)
            {
                binding_layouts[idx] = handle;
                return *this;
            }

            Descriptor& addBindlessLayout(const nvrhi::BindlessLayoutHandle& handle, int idx)
            {
                binding_layouts[idx] = handle;
                return *this;
            }

            friend class ComputePassNode;

        private:
            void resolve(ResourceAllocator& resourceAllocator)
            {
                pipelineDesc.bindingLayouts = binding_layouts;

                for (int i = nvrhi::c_MaxBindingLayouts - 1; i >= 0; --i)
                {
                    if (!pipelineDesc.bindingLayouts[i])
                    {
                        pipelineDesc.bindingLayouts.resize(i);
                    }
                }
            }

            using BindingLayoutVector =
            nvrhi::static_vector<nvrhi::BindingLayoutHandle, nvrhi::c_MaxBindingLayouts>;
            BindingLayoutVector binding_layouts;
        };
    };


    struct FrameGraphRayTracingPass
    {
        struct Descriptor
        {
            Descriptor()
            {
                binding_layouts = BindingLayoutVector(nvrhi::c_MaxBindingLayouts);
            }

            uint8_t samples = 0; // # of samples (0 = unset, default)

            nvrhi::rt::PipelineDesc pipelineDesc;

            Descriptor& addBindingLayout(const nvrhi::BindingLayoutHandle& handle, int idx)
            {
                binding_layouts[idx] = handle;
                return *this;
            }

            Descriptor& addBindlessLayout(const nvrhi::BindlessLayoutHandle& handle, int idx)
            {
                binding_layouts[idx] = handle;
                return *this;
            }

            friend class RayTracingPassNode;

        private:
            void resolve(ResourceAllocator& resourceAllocator)
            {
                pipelineDesc.globalBindingLayouts = binding_layouts;

                for (int i = nvrhi::c_MaxBindingLayouts - 1; i >= 0; --i)
                {
                    if (!pipelineDesc.globalBindingLayouts[i])
                    {
                        pipelineDesc.globalBindingLayouts.resize(i);
                    }
                }
            }

            using BindingLayoutVector =
            nvrhi::static_vector<nvrhi::BindingLayoutHandle, nvrhi::c_MaxBindingLayouts>;
            BindingLayoutVector binding_layouts;
        };
    };

#ifdef RDG_WITH_CUDA
    struct FrameGraphCudaPass
    {
        struct Descriptor
        {
            Descriptor()
            {
            }
        };
    };
#endif

#ifdef RDG_WITH_OPTIX

    struct FrameGraphOptiXPass
    {
        struct Descriptor
        {
            Descriptor()
            {
            }

            friend class OptiXPassNode;

            nvrhi::OptiXPipelineDesc pipeline_desc;

            // Each module can contain several shaders.
            std::vector<nvrhi::OptiXModuleDesc> module_descs;

            std::pair<nvrhi::OptiXProgramGroupDesc, int> ray_gen_group;
            std::vector<std::tuple<nvrhi::OptiXProgramGroupDesc, int, int, int>> hit_group_group;
            std::vector<std::pair<nvrhi::OptiXProgramGroupDesc, int>> miss_group;

           private:
            void resolve(ResourceAllocator& resourceAllocator)
            {
#ifdef _DEBUG  // Enables debug exceptions during optix launches. This may incur significant
               // performance cost and should only be done during development.
                pipeline_desc.pipeline_compile_options.exceptionFlags =
                    OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                    OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

#else
                pipeline_desc.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
                for (int i = 0; i < module_descs.size(); ++i)
                {
                    module_descs[i].pipeline_compile_options =
                        pipeline_desc.pipeline_compile_options;
#ifdef _DEBUG
                    module_descs[i].module_compile_options.optLevel =
                        OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
                    module_descs[i].module_compile_options.debugLevel =
                        OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
                    module_descs[i].module_compile_options.optLevel =
                        OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
                    module_descs[i].module_compile_options.debugLevel =
                        OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif
                }
            }

            OptixShaderFunc raygen_name;
            std::vector<HitGroup> hitgroup_names;
            std::vector<OptixShaderFunc> miss_names;
        };

        uint32_t id = 0;
    };
#endif

}

#undef NVRHI_ENUM_CLASS_FLAG_OPERATORS
