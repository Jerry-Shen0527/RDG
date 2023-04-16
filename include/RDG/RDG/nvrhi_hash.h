#pragma once

#include<nvrhi/nvrhi.h>

#ifdef RDG_WITH_OPTIX
#include <optix.h>
#endif

namespace nvrhi
{
    template<typename T, uint32_t _max_elements>
    inline bool operator==(
        const static_vector<T, _max_elements>& lhs,
        const static_vector<T, _max_elements>& rhs)
    {
        return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }


    inline bool operator==(const VulkanBindingOffsets& lhs, const VulkanBindingOffsets& rhs)
    {
        return lhs.shaderResource == rhs.shaderResource
               && lhs.sampler == rhs.sampler
               && lhs.constantBuffer == rhs.constantBuffer
               && lhs.unorderedAccess == rhs.unorderedAccess;
    }

    inline bool operator!=(const VulkanBindingOffsets& lhs, const VulkanBindingOffsets& rhs)
    {
        return !(lhs == rhs);
    }


    inline bool operator==(const BindingLayoutDesc& lhs, const BindingLayoutDesc& rhs)
    {
        return lhs.visibility == rhs.visibility
               && lhs.registerSpace == rhs.registerSpace
               && lhs.bindings == rhs.bindings
               && lhs.bindingOffsets == rhs.bindingOffsets;
    }

    inline bool operator!=(const BindingLayoutDesc& lhs, const BindingLayoutDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const TextureDesc& lhs, const TextureDesc& rhs)
    {
        bool ret = lhs.width == rhs.width
               && lhs.height == rhs.height
               && lhs.depth == rhs.depth
               && lhs.arraySize == rhs.arraySize
               && lhs.mipLevels == rhs.mipLevels
               && lhs.sampleCount == rhs.sampleCount
               && lhs.sampleQuality == rhs.sampleQuality
               && lhs.format == rhs.format
               && lhs.dimension == rhs.dimension
               && lhs.isRenderTarget == rhs.isRenderTarget
               && lhs.isUAV == rhs.isUAV
               && lhs.isTypeless == rhs.isTypeless
               && lhs.isShadingRateSurface == rhs.isShadingRateSurface
               && lhs.sharedResourceFlags == rhs.sharedResourceFlags
               && lhs.isVirtual == rhs.isVirtual
               && lhs.clearValue == rhs.clearValue
               && lhs.useClearValue == rhs.useClearValue
               && lhs.initialState == rhs.initialState
               && lhs.keepInitialState == rhs.keepInitialState;
#ifdef RDG_WITH_CUDA
        return ret && lhs.mapped_id == rhs.mapped_id;
#endif
    }

    inline bool operator!=(const TextureDesc& lhs, const TextureDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const SamplerDesc& lhs, const SamplerDesc& rhs)
    {
        return lhs.borderColor == rhs.borderColor
               && lhs.maxAnisotropy == rhs.maxAnisotropy
               && lhs.mipBias == rhs.mipBias
               && lhs.minFilter == rhs.minFilter
               && lhs.magFilter == rhs.magFilter
               && lhs.mipFilter == rhs.mipFilter
               && lhs.addressU == rhs.addressU
               && lhs.addressV == rhs.addressV
               && lhs.addressW == rhs.addressW
               && lhs.reductionType == rhs.reductionType;
    }

    inline bool operator!=(const SamplerDesc& lhs, const SamplerDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const RasterState& lhs, const RasterState& rhs)
    {
        return lhs.fillMode == rhs.fillMode
               && lhs.cullMode == rhs.cullMode
               && lhs.frontCounterClockwise == rhs.frontCounterClockwise
               && lhs.depthClipEnable == rhs.depthClipEnable
               && lhs.scissorEnable == rhs.scissorEnable
               && lhs.multisampleEnable == rhs.multisampleEnable
               && lhs.antialiasedLineEnable == rhs.antialiasedLineEnable
               && lhs.depthBias == rhs.depthBias
               && lhs.depthBiasClamp == rhs.depthBiasClamp
               && lhs.slopeScaledDepthBias == rhs.slopeScaledDepthBias
               && lhs.forcedSampleCount == rhs.forcedSampleCount
               && lhs.programmableSamplePositionsEnable == rhs.programmableSamplePositionsEnable
               && lhs.conservativeRasterEnable == rhs.conservativeRasterEnable
               && lhs.quadFillEnable == rhs.quadFillEnable;
    }

    inline bool operator!=(const RasterState& lhs, const RasterState& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(
        const DepthStencilState::StencilOpDesc& lhs,
        const DepthStencilState::StencilOpDesc& rhs)
    {
        return lhs.failOp == rhs.failOp
               && lhs.depthFailOp == rhs.depthFailOp
               && lhs.passOp == rhs.passOp
               && lhs.stencilFunc == rhs.stencilFunc;
    }

    inline bool operator!=(
        const DepthStencilState::StencilOpDesc& lhs,
        const DepthStencilState::StencilOpDesc& rhs)
    {
        return !(lhs == rhs);
    }


    inline bool operator==(const DepthStencilState& lhs, const DepthStencilState& rhs)
    {
        return lhs.depthTestEnable == rhs.depthTestEnable
               && lhs.depthWriteEnable == rhs.depthWriteEnable
               && lhs.depthFunc == rhs.depthFunc
               && lhs.stencilEnable == rhs.stencilEnable
               && lhs.stencilReadMask == rhs.stencilReadMask
               && lhs.stencilWriteMask == rhs.stencilWriteMask
               && lhs.stencilRefValue == rhs.stencilRefValue
               && lhs.frontFaceStencil == rhs.frontFaceStencil
               && lhs.backFaceStencil == rhs.backFaceStencil;
    }

    inline bool operator!=(const DepthStencilState& lhs, const DepthStencilState& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const RenderState& lhs, const RenderState& rhs)
    {
        return lhs.blendState == rhs.blendState
               && lhs.depthStencilState == rhs.depthStencilState
               && lhs.rasterState == rhs.rasterState
               && lhs.singlePassStereo == rhs.singlePassStereo;
    }

    inline bool operator!=(const RenderState& lhs, const RenderState& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const GraphicsPipelineDesc& lhs, const GraphicsPipelineDesc& rhs)
    {
        return lhs.primType == rhs.primType
               && lhs.patchControlPoints == rhs.patchControlPoints
               && lhs.inputLayout == rhs.inputLayout
               && lhs.VS == rhs.VS
               && lhs.HS == rhs.HS
               && lhs.DS == rhs.DS
               && lhs.GS == rhs.GS
               && lhs.PS == rhs.PS
               && lhs.renderState == rhs.renderState
               && lhs.shadingRateState == rhs.shadingRateState
               && lhs.bindingLayouts == rhs.bindingLayouts;
    }

    inline bool operator!=(const GraphicsPipelineDesc& lhs, const GraphicsPipelineDesc& rhs)
    {
        return !(lhs == rhs);
    }


    inline bool operator==(const FramebufferAttachment& lhs, const FramebufferAttachment& rhs)
    {
        return lhs.texture == rhs.texture
               && lhs.subresources == rhs.subresources
               && lhs.format == rhs.format
               && lhs.isReadOnly == rhs.isReadOnly;
    }

    inline bool operator!=(const FramebufferAttachment& lhs, const FramebufferAttachment& rhs)
    {
        return !(lhs == rhs);
    }


    inline bool operator==(const FramebufferDesc& lhs, const FramebufferDesc& rhs)
    {
        return lhs.colorAttachments == rhs.colorAttachments
               && lhs.depthAttachment == rhs.depthAttachment
               && lhs.shadingRateAttachment == rhs.shadingRateAttachment;
    }

    inline bool operator!=(const FramebufferDesc& lhs, const FramebufferDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const ComputePipelineDesc& lhs, const ComputePipelineDesc& rhs)
    {
        return lhs.CS == rhs.CS
               && lhs.bindingLayouts == rhs.bindingLayouts;
    }

    inline bool operator!=(const ComputePipelineDesc& lhs, const ComputePipelineDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const BindlessLayoutDesc& lhs, const BindlessLayoutDesc& rhs)
    {
        return lhs.visibility == rhs.visibility
               && lhs.firstSlot == rhs.firstSlot
               && lhs.maxCapacity == rhs.maxCapacity
               && lhs.registerSpaces == rhs.registerSpaces;
    }

    inline bool operator!=(const BindlessLayoutDesc& lhs, const BindlessLayoutDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const BufferDesc& lhs, const BufferDesc& rhs)
    {
        return lhs.byteSize == rhs.byteSize
               && lhs.structStride == rhs.structStride
               && lhs.maxVersions == rhs.maxVersions
               && lhs.debugName == rhs.debugName
               && lhs.format == rhs.format
               && lhs.canHaveUAVs == rhs.canHaveUAVs
               && lhs.canHaveTypedViews == rhs.canHaveTypedViews
               && lhs.canHaveRawViews == rhs.canHaveRawViews
               && lhs.isVertexBuffer == rhs.isVertexBuffer
               && lhs.isIndexBuffer == rhs.isIndexBuffer
               && lhs.isConstantBuffer == rhs.isConstantBuffer
               && lhs.isDrawIndirectArgs == rhs.isDrawIndirectArgs
               && lhs.isAccelStructBuildInput == rhs.isAccelStructBuildInput
               && lhs.isAccelStructStorage == rhs.isAccelStructStorage
               && lhs.isVolatile == rhs.isVolatile
               && lhs.isVirtual == rhs.isVirtual
               && lhs.initialState == rhs.initialState
               && lhs.keepInitialState == rhs.keepInitialState
               && lhs.cpuAccess == rhs.cpuAccess
               && lhs.sharedResourceFlags == rhs.sharedResourceFlags;
    }

    inline bool operator!=(const BufferDesc& lhs, const BufferDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const CommandListParameters& lhs, const CommandListParameters& rhs)
    {
        return lhs.enableImmediateExecution == rhs.enableImmediateExecution
               && lhs.uploadChunkSize == rhs.uploadChunkSize
               && lhs.scratchChunkSize == rhs.scratchChunkSize
               && lhs.scratchMaxMemory == rhs.scratchMaxMemory
               && lhs.queueType == rhs.queueType;
    }

    inline bool operator!=(const CommandListParameters& lhs, const CommandListParameters& rhs)
    {
        return !(lhs == rhs);
    }

    namespace rt
    {
        inline bool operator==(const PipelineHitGroupDesc& lhs, const PipelineHitGroupDesc& rhs)
        {
            return lhs.exportName == rhs.exportName
                   && lhs.closestHitShader == rhs.closestHitShader
                   && lhs.anyHitShader == rhs.anyHitShader
                   && lhs.intersectionShader == rhs.intersectionShader
                   && lhs.bindingLayout == rhs.bindingLayout
                   && lhs.isProceduralPrimitive == rhs.isProceduralPrimitive;
        }

        inline bool operator!=(const PipelineHitGroupDesc& lhs, const PipelineHitGroupDesc& rhs)
        {
            return !(lhs == rhs);
        }

        inline bool operator==(const PipelineShaderDesc& lhs, const PipelineShaderDesc& rhs)
        {
            return lhs.exportName == rhs.exportName
                   && lhs.shader == rhs.shader
                   && lhs.bindingLayout == rhs.bindingLayout;
        }

        inline bool operator!=(const PipelineShaderDesc& lhs, const PipelineShaderDesc& rhs)
        {
            return !(lhs == rhs);
        }

        inline bool operator==(const PipelineDesc& lhs, const PipelineDesc& rhs)
        {
            return lhs.shaders == rhs.shaders
                   && lhs.hitGroups == rhs.hitGroups
                   && lhs.globalBindingLayouts == rhs.globalBindingLayouts
                   && lhs.maxPayloadSize == rhs.maxPayloadSize
                   && lhs.maxAttributeSize == rhs.maxAttributeSize
                   && lhs.maxRecursionDepth == rhs.maxRecursionDepth;
        }

        inline bool operator!=(const PipelineDesc& lhs, const PipelineDesc& rhs)
        {
            return !(lhs == rhs);
        }


        inline bool operator==(const GeometryTriangles& lhs, const GeometryTriangles& rhs)
        {
            return lhs.indexBuffer == rhs.indexBuffer && lhs.vertexBuffer == rhs.vertexBuffer &&
                   lhs.indexFormat == rhs.indexFormat && lhs.vertexFormat == rhs.vertexFormat &&
                   lhs.indexOffset == rhs.indexOffset && lhs.vertexOffset == rhs.vertexOffset &&
                   lhs.indexCount == rhs.indexCount && lhs.vertexCount == rhs.vertexCount &&
                   lhs.vertexStride == rhs.vertexStride &&
                   lhs.opacityMicromap == rhs.opacityMicromap &&
                   lhs.ommIndexBuffer == rhs.ommIndexBuffer &&
                   lhs.ommIndexBufferOffset == rhs.ommIndexBufferOffset &&
                   lhs.ommIndexFormat == rhs.ommIndexFormat &&
                   lhs.pOmmUsageCounts == rhs.pOmmUsageCounts &&
                   lhs.numOmmUsageCounts == rhs.numOmmUsageCounts;
        }

        inline bool operator!=(const GeometryTriangles& lhs, const GeometryTriangles& rhs)
        {
            return !(lhs == rhs);
        }

        inline bool operator==(const GeometryDesc& lhs, const GeometryDesc& rhs)
        {
            return lhs.geometryData.triangles == rhs.geometryData.triangles &&
                   lhs.useTransform == rhs.useTransform && lhs.flags == rhs.flags &&
                   lhs.geometryType == rhs.geometryType && memcmp(
                       lhs.transform,
                       rhs.transform,
                       sizeof(nvrhi::rt::AffineTransform));
        }

        inline bool operator!=(const GeometryDesc& lhs, const GeometryDesc& rhs)
        {
            return !(lhs == rhs);
        }


        inline bool operator==(const AccelStructDesc& lhs, const AccelStructDesc& rhs)
        {
            return lhs.topLevelMaxInstances == rhs.topLevelMaxInstances &&
                   lhs.bottomLevelGeometries == rhs.bottomLevelGeometries &&
                   lhs.buildFlags == rhs.buildFlags && lhs.debugName == rhs.debugName &&
                   lhs.trackLiveness == rhs.trackLiveness && lhs.isTopLevel == rhs.isTopLevel &&
                   lhs.isVirtual == rhs.isVirtual;
        }

        inline bool operator!=(const AccelStructDesc& lhs, const AccelStructDesc& rhs)
        {
            return !(lhs == rhs);
        }
    }

#ifdef NVRHI_WITH_CUDA
    inline bool operator==(const CudaLinearBufferDesc& lhs, const CudaLinearBufferDesc& rhs)
    {
        return lhs.size == rhs.size && lhs.element_size == rhs.element_size &&
               lhs.bufferType == rhs.bufferType && lhs.mapped_id == rhs.mapped_id;
    }

    inline bool operator!=(const CudaLinearBufferDesc& lhs, const CudaLinearBufferDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const CudaSurfaceObjectDesc& lhs, const CudaSurfaceObjectDesc& rhs)
    {
        return lhs.width == rhs.width && lhs.height == rhs.height &&
               lhs.element_size == rhs.element_size && lhs.bufferType == rhs.bufferType &&
               lhs.mapped_id == rhs.mapped_id;
    }

    inline bool operator!=(const CudaSurfaceObjectDesc& lhs, const CudaSurfaceObjectDesc& rhs)
    {
        return !(lhs == rhs);
    }
#endif

#ifdef NVRHI_WITH_OPTIX
    inline bool operator==(const OptixBuiltinISOptions& lhs, const OptixBuiltinISOptions& rhs)
    {
        return lhs.builtinISModuleType == rhs.builtinISModuleType &&
               lhs.usesMotionBlur == rhs.usesMotionBlur && lhs.buildFlags == rhs.buildFlags &&
               lhs.curveEndcapFlags == rhs.curveEndcapFlags;
    }

    inline bool operator==(
        const OptixModuleCompileOptions& lhs,
        const OptixModuleCompileOptions& rhs)
    {
        return lhs.maxRegisterCount == rhs.maxRegisterCount && lhs.optLevel == rhs.optLevel &&
               lhs.debugLevel == rhs.debugLevel && lhs.boundValues == rhs.boundValues &&
               lhs.numBoundValues == rhs.numBoundValues &&
               lhs.numPayloadTypes == rhs.numPayloadTypes && lhs.payloadTypes == rhs.payloadTypes;
    }

    inline bool operator==(
        const OptixPipelineCompileOptions& lhs,
        const OptixPipelineCompileOptions& rhs)
    {
        return lhs.usesMotionBlur == rhs.usesMotionBlur &&
               lhs.traversableGraphFlags == rhs.traversableGraphFlags &&
               lhs.numPayloadValues == rhs.numPayloadValues &&
               lhs.numAttributeValues == rhs.numAttributeValues &&
               lhs.exceptionFlags == rhs.exceptionFlags &&
               lhs.pipelineLaunchParamsVariableName == rhs.pipelineLaunchParamsVariableName &&
               lhs.usesPrimitiveTypeFlags == rhs.usesPrimitiveTypeFlags;
    }

    inline bool operator==(const OptiXModuleDesc& lhs, const OptiXModuleDesc& rhs)
    {
        return lhs.module_compile_options == rhs.module_compile_options &&
               lhs.pipeline_compile_options == rhs.pipeline_compile_options &&
               lhs.builtinISOptions == rhs.builtinISOptions && lhs.ptx == rhs.ptx;
    }

    inline bool operator!=(const OptiXModuleDesc& lhs, const OptiXModuleDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const OptixPipelineLinkOptions& lhs, const OptixPipelineLinkOptions& rhs)
    {
        return lhs.maxTraceDepth == rhs.maxTraceDepth;
    }

    inline bool operator==(
        const OptixProgramGroupSingleModule& lhs,
        const OptixProgramGroupSingleModule& rhs)
    {
        return lhs.module == rhs.module && lhs.entryFunctionName == rhs.entryFunctionName;
    }

    inline bool operator==(const OptixProgramGroupDesc& lhs, const OptixProgramGroupDesc& rhs)
    {
        return lhs.kind == rhs.kind && lhs.flags == rhs.flags && lhs.raygen == rhs.raygen;
    }

    inline bool operator==(const OptixProgramGroupOptions& lhs, const OptixProgramGroupOptions& rhs)
    {
        return lhs.payloadType == rhs.payloadType;
    }


    inline bool operator==(const OptiXPipelineDesc& lhs, const OptiXPipelineDesc& rhs)
    {
        return lhs.pipeline_compile_options == rhs.pipeline_compile_options &&
               lhs.pipeline_link_options == rhs.pipeline_link_options;
    }

    inline bool operator!=(const OptiXPipelineDesc& lhs, const OptiXPipelineDesc& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator==(const OptiXProgramGroupDesc& lhs, const OptiXProgramGroupDesc& rhs)
    {
        return lhs.program_group_options == rhs.program_group_options &&
               lhs.prog_group_desc == rhs.prog_group_desc;
    }

    inline bool operator!=(const OptiXProgramGroupDesc& lhs, const OptiXProgramGroupDesc& rhs)
    {
        return !(lhs == rhs);
    }

#endif
}
