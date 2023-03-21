#include <RDG/RDG/ResourceAllocator.h>

#include <iostream>


/*
 * Copyright (C) 2019 The Android Open Source Project
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

namespace Furnace
{
    ResourceAllocator::ResourceAllocator(
        nvrhi::DeviceHandle device)
    noexcept
        : mDevice(device)
    {
    }

    void ResourceAllocator::dump(auto& cache, bool brief, uint32_t cacheSize) const noexcept
    {
        std::cerr << "# entries=" << cache.size() << ", sz=" << cacheSize / float(
                1u << 20u)
            << " MiB" << std::endl;
        if (!brief)
        {
            for (const auto& it : cache)
            {
                auto w = it.first.width;
                auto h = it.first.height;
                auto f = nvrhi::getFormatInfo(it.first.format).bytesPerBlock;
                std::cerr << it.first.debugName << ": w=" << w << ", h=" << h << ", f=" << f
                    << ", sz=" << it.second.size / float(1u << 20u) << std::endl;
            }
        }
    }
}
