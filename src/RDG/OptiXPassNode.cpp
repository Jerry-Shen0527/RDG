#ifdef RDG_WITH_OPTIX

#include <optix_stubs.h>
#include"PassNode.h"
#include "RDG/RDG/FrameGraph.h"
#include <RDG/RDG/DescTrait.h>
#include <iomanip>
#include "RDG/CUDA/CUDAException.h"

namespace Furnace
{
    template<typename IntegerType>
    IntegerType roundUp(IntegerType x, IntegerType y)
    {
        return ((x + y - 1) / y) * y;
    }

    template<typename T>
    struct SbtRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    using RayGenSbtRecord = SbtRecord<int>;
    using HitGroupSbtRecord = SbtRecord<int>;
    using MissSbtRecord = SbtRecord<int>;


    void OptiXPassNode::OptiXPassData::devirtualize(
        FrameGraph& fg,
        ResourceAllocator& resourceAllocator) noexcept
    {
        // Prepare the modules

        descriptor.resolve(resourceAllocator);

        {
            for (int i = 0; i < descriptor.module_descs.size(); ++i)
            {
                solidModules.push_back(
                    resourceAllocator.create(
                        descriptor.module_descs[i]));
            }
        }

        auto raygen_desc = descriptor.ray_gen_group.first;
        raygen_desc.prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        auto module_idx = descriptor.ray_gen_group.second;
        solid_raygen_group = resourceAllocator.create(raygen_desc, solidModules[module_idx]);

        const int hitgroup_count = descriptor.hit_group_group.size();
        assert(hitgroup_count > 0);
        solid_hitgroup_group.reserve(hitgroup_count);

        for (int i = 0; i < hitgroup_count; ++i)
        {
            auto group = descriptor.hit_group_group[i];

            nvrhi::OptiXProgramGroupDesc desc = std::get<0>(group);
            desc.prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            auto modules = std::make_tuple(
                solidModules[std::get<1>(group)],
                solidModules[std::get<2>(group)],
                solidModules[std::get<3>(group)]);

            solid_hitgroup_group.push_back(resourceAllocator.create(desc, modules));
        }

        const int miss_count = descriptor.miss_group.size();

        solid_miss_group.reserve(miss_count);
        for (int i = 0; i < miss_count; ++i)
        {
            auto group = descriptor.miss_group[i];
            group.first.prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            nvrhi::OptiXProgramGroupDesc desc = group.first;
            auto module = solidModules[group.second];

            solid_miss_group.push_back(resourceAllocator.create(desc, module));
        }

        std::vector<nvrhi::OptiXProgramGroupHandle> program_groups;
        program_groups.push_back(solid_raygen_group);
        program_groups.insert(
            program_groups.end(),
            solid_hitgroup_group.begin(),
            solid_hitgroup_group.end());
        program_groups.insert(
            program_groups.end(),
            solid_miss_group.begin(),
            solid_miss_group.end());

        solid_handle = resourceAllocator.create(descriptor.pipeline_desc, program_groups);

        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        raygen_record =
            resourceAllocator.create(nvrhi::CudaLinearBufferDesc{ 1, raygen_record_size });
        void* d_raygen_record = raygen_record->GetGPUAddress();
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(solid_raygen_group->getProgramGroup(), &rg_sbt));
        CUDA_CHECK(
            cudaMemcpy(
                reinterpret_cast<void*>(d_raygen_record),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice));

        int hitgroupRecordStrideInBytes =
            roundUp<int>(sizeof(HitGroupSbtRecord), OPTIX_SBT_RECORD_ALIGNMENT);

        const int hitgroup_record_size = hitgroupRecordStrideInBytes * hitgroup_count;

        hitgroup_record = resourceAllocator.create(
            nvrhi::CudaLinearBufferDesc{ hitgroup_count, hitgroupRecordStrideInBytes });
        void* d_hitgroup_record = hitgroup_record->GetGPUAddress();
        std::vector<HitGroupSbtRecord> hg_sbts(hitgroup_count);

        for (int i = 0; i < hitgroup_count; ++i)
        {
            OPTIX_CHECK(
                optixSbtRecordPackHeader(solid_hitgroup_group[i]->getProgramGroup(), &hg_sbts[i]));
        }

        CUDA_CHECK(
            cudaMemcpy(
                reinterpret_cast<void*>(d_hitgroup_record),
                hg_sbts.data(),
                hitgroup_record_size,
                cudaMemcpyHostToDevice));

        if (miss_count > 0)
        {
            int missRecordStrideInBytes =
                roundUp<size_t>(sizeof(MissSbtRecord), OPTIX_SBT_RECORD_ALIGNMENT);

            int miss_record_size = missRecordStrideInBytes * miss_count;

            miss_record = resourceAllocator.create(
                nvrhi::CudaLinearBufferDesc{ miss_count, missRecordStrideInBytes });

            void* d_miss_record = miss_record->GetGPUAddress();

            std::vector<MissSbtRecord> ms_sbts(miss_count);
            for (int i = 0; i < miss_count; ++i)
            {
                // currently, do nothing.
                OPTIX_CHECK(
                    optixSbtRecordPackHeader(solid_miss_group[i]->getProgramGroup(), &ms_sbts[i]));
            }

            CUDA_CHECK(
                cudaMemcpy(
                    reinterpret_cast<void*>(d_miss_record),
                    ms_sbts.data(),
                    miss_record_size,
                    cudaMemcpyHostToDevice));

            sbt.missRecordBase = CUdeviceptr(d_miss_record);
            sbt.missRecordStrideInBytes = missRecordStrideInBytes;
            sbt.missRecordCount = miss_count;
        }
        else
        {
            sbt.missRecordBase = 0;
        }

        sbt.raygenRecord = CUdeviceptr(d_raygen_record);
        sbt.hitgroupRecordBase = CUdeviceptr(d_hitgroup_record);
        sbt.hitgroupRecordStrideInBytes = hitgroupRecordStrideInBytes;
        sbt.hitgroupRecordCount = hitgroup_count;
    }

    void OptiXPassNode::OptiXPassData::destroy(ResourceAllocator& resourceAllocator) noexcept
    {
        for (int i = 0; i < solidModules.size(); ++i)
        {
            resourceAllocator.destroy(solidModules[i]);
        }
        solidModules.clear();

        for (int i = 0; i < solid_hitgroup_group.size(); ++i)
        {
            resourceAllocator.destroy(solid_hitgroup_group[i]);
        }
        solid_hitgroup_group.clear();

        for (int i = 0; i < solid_miss_group.size(); ++i)
        {
            resourceAllocator.destroy(solid_miss_group[i]);
        }
        solid_miss_group.clear();

        resourceAllocator.destroy(solid_raygen_group);
        resourceAllocator.destroy(raygen_record);
        resourceAllocator.destroy(hitgroup_record);
        resourceAllocator.destroy(miss_record);
        resourceAllocator.destroy(solid_handle);
    }


    void OptiXPassNode::execute(
        const FrameGraphResources& resources,
        nvrhi::DeviceHandle& device) noexcept
    {
        mOptiXData.devirtualize(mFrameGraph, mFrameGraph.getResourceAllocator());

        mPassBase->execute(resources, device);

        mOptiXData.destroy(mFrameGraph.getResourceAllocator());
    }

    void OptiXPassNode::resolve() noexcept
    {
    }


    void OptiXPassNode::declareOptiXPass(
        FrameGraph& fg,
        FrameGraph::Builder& builder,
        const char* name,
        const FrameGraphOptiXPass::Descriptor& descriptor)
    {
        mOptiXData.name = name;
        mOptiXData.descriptor = descriptor;
    }

    const OptiXPassNode::OptiXPassData* OptiXPassNode::getOptiXPassData() const noexcept
    {
        return &mOptiXData;
    }
}


#endif