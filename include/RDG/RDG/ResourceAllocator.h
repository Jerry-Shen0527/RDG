#pragma once

#include "nvrhi_hash.h"
#include <vector>

#include <algorithm>
#include <iterator>
#include <memory>

#include "DescTrait.h"



/*
 * Copyright from filament
 *
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
    class ResourceAllocator
    {
#define CACHE_NAME(RESOURCE)   m##RESOURCE##Cache
#define INUSE_NAME(RESOURCE)   mInUse##RESOURCE
#define PAYLOAD_NAME(RESOURCE) RESOURCE##CachePayload
#define CACHE_SIZE(RESOURCE)   m##RESOURCE##CacheSize

#define JUDGE_RESOURCE(NAMESPACE, RSC) \
    if constexpr (                                          \
        std::is_same_v<NAMESPACE::RSC##Handle, RESOURCE> || \
        std::is_same_v<NAMESPACE::I##RSC*, RESOURCE>)

#define RESOLVE_DESTROY(NAMESPACE, RESOURCE)                             \
    NAMESPACE::RESOURCE##Handle h = NAMESPACE::RESOURCE##Handle(handle); \
    PAYLOAD_NAME(RESOURCE) payload{ h, mAge, 0 };                        \
    resolveCacheDestroy(                                                 \
        h, CACHE_SIZE(RESOURCE), payload, CACHE_NAME(RESOURCE), INUSE_NAME(RESOURCE));


    public:
        explicit ResourceAllocator(nvrhi::DeviceHandle device) noexcept;

#define CHECK_EMPTY(RESOURCE)             \
    assert(!CACHE_NAME(RESOURCE).size()); \
    assert(!INUSE_NAME(RESOURCE).size());

        ~ResourceAllocator() noexcept
        {
            {
                terminate();
                DRJIT_MAP(CHECK_EMPTY, RESOURCE_LIST)
            }
        }


#define CLEAR_CACHE(RESOURCE)                                                       \
    assert(!INUSE_NAME(RESOURCE).size());                                           \
    for (auto it = CACHE_NAME(RESOURCE).begin(); it != CACHE_NAME(RESOURCE).end();) \
    {                                                                               \
        it->second.handle = nullptr;                                                \
        it = CACHE_NAME(RESOURCE).erase(it);                                        \
    }

        void terminate() noexcept
        {
            DRJIT_MAP(CLEAR_CACHE, RESOURCE_LIST)
        }


        void dump(auto& cache, bool brief, uint32_t cacheSize) const noexcept;

#define FOREACH_DESTROY(NAMESPACE, RESOURCE) \
    JUDGE_RESOURCE(NAMESPACE, RESOURCE)      \
    {                                        \
        RESOLVE_DESTROY(NAMESPACE, RESOURCE) \
    }

        template<typename RESOURCE>
        void destroy(RESOURCE& handle) noexcept
        {
            if constexpr (mEnabled)
            {
#define WRAPPED_MACRO FOREACH_DESTROY
                MAP_TO_ALL_NAMESPACE
            }
            else
            {
                handle = nullptr;
            }
        }

#define GC_TYPE(NAMESPACE,RSC)\
            gc_type<NAMESPACE::RSC##Handle>(CACHE_SIZE(RSC), CACHE_NAME(RSC));


        void gc() noexcept
        {
#define WRAPPED_MACRO GC_TYPE
            MAP_TO_ALL_NAMESPACE
        }


#define RESOLVE_CREATE(RESOURCE) \
    resolveCacheCreate(          \
        handle, desc, CACHE_SIZE(RESOURCE), CACHE_NAME(RESOURCE), INUSE_NAME(RESOURCE), rest...);

#define FOREACH_CREATE(NAMESPACE, RESOURCE) \
    JUDGE_RESOURCE(NAMESPACE, RESOURCE)      \
    {                                        \
        RESOLVE_CREATE(RESOURCE)            \
    }

        template<typename DESC, typename RESOURCE = resc<DESC>, typename... Args>
        RESOURCE create(const DESC& desc, Args&&... rest)
        {
            RESOURCE handle;

            if constexpr (mEnabled)
            {
#define WRAPPED_MACRO FOREACH_CREATE
                MAP_TO_ALL_NAMESPACE
            }
            else
            {
                handle = create_resource(desc, std::forward<Args>(rest)...);
            }
            return handle;
        }

#define Container(NAMESPACE, RESOURCE)                                                \
    struct PAYLOAD_NAME(RESOURCE)                                                     \
    {                                                                                 \
        NAMESPACE::RESOURCE##Handle handle;                                           \
        size_t age = 0;                                                               \
        uint32_t size = 0;                                                            \
    };                                                                                \
    using RESOURCE##CacheContainer =                                                  \
        AssociativeContainer<NAMESPACE::RESOURCE##Desc, RESOURCE##CachePayload>;      \
    using RESOURCE##InUseContainer =                                                  \
        AssociativeContainer<NAMESPACE::RESOURCE##Handle, NAMESPACE::RESOURCE##Desc>; \
    RESOURCE##CacheContainer CACHE_NAME(RESOURCE);                                    \
    RESOURCE##InUseContainer INUSE_NAME(RESOURCE);                                    \
    uint32_t CACHE_SIZE(RESOURCE) = 0;

#define PURGE(NAMESPACE, RESOURCE)                                                          \
    RESOURCE##CacheContainer::iterator purge(const RESOURCE##CacheContainer::iterator& pos) \
    {                                                                                       \
        pos->second.handle = nullptr;                                                       \
        m##RESOURCE##CacheSize -= pos->second.size;                                         \
        return CACHE_NAME(RESOURCE).erase(pos);                                             \
    }

        // Not recommended
        [[nodiscard]] nvrhi::DeviceHandle GetNvrhiDevice() const
        {
            return mDevice;
        }

    private:
#define CREATE_CONCRETE_NVRHI(RESOURCE)    \
    JUDGE_RESOURCE(nvrhi, RESOURCE)         \
    {                                           \
        return mDevice->create##RESOURCE(desc,rest...); \
    }
#define CREATE_CONCRETE_NVRHI_RT(RESOURCE)                  \
    JUDGE_RESOURCE(nvrhi::rt, RESOURCE)                      \
    {                                                    \
        return mDevice->create##RESOURCE(desc, rest...); \
    }

        template<typename RESOURCE, typename... Args>
        RESOURCE create_resource(const desc<RESOURCE>& desc, Args&&... rest)
        {
            DRJIT_MAP(CREATE_CONCRETE_NVRHI, NVRHI_RESOURCE_LIST)

            // This is kind of awkward...... but just do this.
            if constexpr (std::is_same_v<nvrhi::rt::PipelineHandle, RESOURCE> || std::is_same_v<
                              nvrhi::rt::IPipeline*, RESOURCE>)
            {
                return mDevice->createRayTracingPipeline(desc, rest...);
            }

                        // This is kind of awkward...... but just do this.
            if constexpr (
                std::is_same_v<nvrhi::rt::ShaderTableHandle, RESOURCE> ||
                std::is_same_v<nvrhi::rt::IShaderTable*, RESOURCE>)
            {
                return desc->createShaderTable();
            }
            CREATE_CONCRETE_NVRHI_RT(AccelStruct)
        }


        template<typename RESOURCE>
        void
        resolveCacheCreate(
            RESOURCE& handle,
            auto& desc,
            auto& cacheSize,
            auto&& cache,
            auto&& inUseCache,
            auto&&... rest)
        {
            auto it = cache.find(desc);
            if (it != cache.end())
            {
                // we do, move the entry to the in-use list, and remove from the cache
                handle = it->second.handle;
                cacheSize -= it->second.size;
                cache.erase(it);
            }
            else
            {
                handle = create_resource<RESOURCE>(desc, rest...);
            }
            inUseCache.emplace(handle, desc);
        }


        template<typename RESOURCE>
        uint32_t calcSize(desc<RESOURCE>& key)
        {
            JUDGE_RESOURCE(nvrhi, Texture)
            {
                auto info = nvrhi::getFormatInfo(key.format);
                return key.width * key.height * key.depth * info.bytesPerBlock * info.blockSize;
            }

            return 0;
        }

        template<typename RESOURCE>
        void resolveCacheDestroy(
            RESOURCE& handle,
            auto& cacheSize,
            auto& cachePayload,
            auto&& cache,
            auto&& inUseCache)
        {
            // find the texture in the in-use list (it must be there!)
            auto it = inUseCache.find(handle);
            assert(it != inUseCache.end());

            // move it to the cache
            auto key = it->second;

            cachePayload.size = calcSize<RESOURCE>(key);

            //cache.emplace(key, CachePayload{ handle, mAge, size });
            cache.emplace(key, cachePayload);
            cacheSize += cachePayload.size;

            // remove it from the in-use list
            inUseCache.erase(it);
        }

        template<typename RESOURCE>
        void gc_type(auto& cacheSize, auto&& cache_in)
        {
            // this is called regularly -- usually once per frame of each Renderer

            // increase our age
            const size_t age = mAge++;

            // Purging strategy:
            //  - remove entries that are older than a certain age
            //      - remove only one entry per gc(),
            //      - unless we're at capacity
            // - remove LRU entries until we're below capacity

            for (auto it = cache_in.begin(); it != cache_in.end();)
            {
                const size_t ageDiff = age - it->second.age;
                if (ageDiff >= CACHE_MAX_AGE)
                {
                    it = purge(it);
                    if (cacheSize < CACHE_CAPACITY)
                    {
                        // if we're not at capacity, only purge a single entry per gc, trying to
                        // avoid a burst of work.
                        break;
                    }
                }
                else
                {
                    ++it;
                }
            }

            if ((cacheSize >= CACHE_CAPACITY))
            {
                // make a copy of our CacheContainer to a vector
                using ContainerType = std::remove_cvref_t<decltype(cache_in)>;
                using Vector = std::vector<std::pair<
                    typename ContainerType::key_type, typename ContainerType::value_type>>;
                auto cache = Vector();
                std::copy(
                    cache_in.begin(),
                    cache_in.end(),
                    std::back_insert_iterator<Vector>(cache));

                // sort by least recently used
                std::sort(
                    cache.begin(),
                    cache.end(),
                    [](const auto& lhs, const auto& rhs)
                    {
                        return lhs.second.age < rhs.second.age;
                    });

                // now remove entries until we're at capacity
                auto curr = cache.begin();
                while (cacheSize >= CACHE_CAPACITY)
                {
                    // by construction this entry must exist
                    purge(cache_in.find(curr->first));
                    ++curr;
                }

                // Since we're sorted already, reset the oldestAge of the whole system
                size_t oldestAge = cache.front().second.age;
                for (auto& it : cache_in)
                {
                    it.second.age -= oldestAge;
                }
                mAge -= oldestAge;
            }
            // if (mAge % 60 == 0) dump();
        }

        // TODO: these should be settings of the engine
        static constexpr size_t CACHE_CAPACITY = 64u << 20u; // 64 MiB
        static constexpr size_t CACHE_MAX_AGE = 30u;


        template<typename T>
        struct Hasher
        {
            std::size_t operator()(const T& s) const noexcept
            {
                return hash_value(s);
            }
        };

        template<typename T>
        struct Hasher<nvrhi::RefCountPtr<T>>
        {
            auto operator()(const nvrhi::RefCountPtr<T>& s) const noexcept
            {
                return s.Get();
            }
        };

        void dump(bool brief = false, uint32_t cacheSize = 0) const noexcept;

        template<typename Key, typename Value, typename Hasher = Hasher<Key>>
        class AssociativeContainer
        {
            // We use a std::vector instead of a std::multimap because we don't expect many items
            // in the cache and std::multimap generates tons of code. std::multimap starts getting
            // significantly better around 1000 items.
            using Container = std::vector<std::pair<Key, Value>>;
            Container mContainer;

        public:
            AssociativeContainer();
            ~AssociativeContainer() noexcept;
            using iterator = typename Container::iterator;
            using const_iterator = typename Container::const_iterator;
            using key_type = typename Container::value_type::first_type;
            using value_type = typename Container::value_type::second_type;

            size_t size() const
            {
                return mContainer.size();
            }

            iterator begin()
            {
                return mContainer.begin();
            }

            const_iterator begin() const
            {
                return mContainer.begin();
            }

            iterator end()
            {
                return mContainer.end();
            }

            const_iterator end() const
            {
                return mContainer.end();
            }

            iterator erase(iterator it);
            const_iterator find(const key_type& key) const;
            iterator find(const key_type& key);
            template<typename... ARGS>
            void emplace(ARGS&&... args);
        };

#define CONTAINER_RELATED(NAMESPACE, RESOURCE) \
    Container(NAMESPACE, RESOURCE);            \
    PURGE(NAMESPACE, RESOURCE)

#define WRAPPED_MACRO CONTAINER_RELATED
        MAP_TO_ALL_NAMESPACE

        nvrhi::DeviceHandle mDevice;
        size_t mAge = 0;
        static constexpr bool mEnabled = true;
    };


    template<typename K, typename V, typename H>
    ResourceAllocator::AssociativeContainer<K, V, H>::AssociativeContainer()
    {
        mContainer.reserve(128);
    }

    template<typename K, typename V, typename H>

    ResourceAllocator::AssociativeContainer<K, V, H>::~AssociativeContainer() noexcept
    {
    }

    template<typename K, typename V, typename H>
    typename ResourceAllocator::AssociativeContainer<K, V, H>::iterator
    ResourceAllocator::AssociativeContainer<K, V, H>::erase(iterator it)
    {
        return mContainer.erase(it);
    }

    template<typename K, typename V, typename H>
    typename ResourceAllocator::AssociativeContainer<K, V, H>::const_iterator
    ResourceAllocator::AssociativeContainer<K, V, H>::find(const key_type& key) const
    {
        return const_cast<AssociativeContainer*>(this)->find(key);
    }

    template<typename K, typename V, typename H>
    typename ResourceAllocator::AssociativeContainer<K, V, H>::iterator
    ResourceAllocator::AssociativeContainer<K, V, H>::find(const key_type& key)
    {
        return std::find_if(
            mContainer.begin(),
            mContainer.end(),
            [&key](const auto& v)
            {
                return v.first == key;
            });
    }

    template<typename K, typename V, typename H>
    template<typename... ARGS>
    void ResourceAllocator::AssociativeContainer<K, V, H>::emplace(ARGS&&... args)
    {
        mContainer.emplace_back(std::forward<ARGS>(args)...);
    }
}  // namespace Furnace

