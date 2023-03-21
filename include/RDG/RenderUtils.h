#pragma once
#include "donut/engine/Scene.h"
#include "nvrhi/nvrhi.h"

namespace Furnace
{
    nvrhi::rt::AccelStructHandle CreateAccelStructs(
        std::shared_ptr<donut::engine::Scene> m_Scene,
        nvrhi::DeviceHandle device,
        nvrhi::CommandListHandle commandList);

    void BuildTLAS(
        nvrhi::rt::AccelStructHandle handle,
        std::shared_ptr<donut::engine::Scene> m_Scene,
        nvrhi::CommandListHandle commandList,
        uint32_t frameIndex);

    inline nvrhi::TextureDesc defaultRenderTarget(
        donut::math::int2 size,
        const nvrhi::Format& format)
    {
        nvrhi::TextureDesc desc;
        desc.width = size.x;
        desc.height = size.y;
        desc.initialState = nvrhi::ResourceStates::RenderTarget;
        desc.isRenderTarget = true;
        desc.useClearValue = false;
        // desc.clearValue = nvrhi::Color(0.f);
        desc.sampleCount = 1;
        desc.dimension = nvrhi::TextureDimension::Texture2D;
        desc.keepInitialState = true;
        desc.isTypeless = false;
        desc.isUAV = true;
        desc.mipLevels = 1;

        desc.format = format;

        return desc;
    }
}
