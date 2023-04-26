#include <iomanip>
#include <iostream>

#ifdef RDG_WITH_OPTIX
#include "RDG/CUDA/CUDAException.h"

#include"optix_function_table_definition.h"

#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <optix.h>
#include <cuda_runtime_api.h>

#include"nvrhi/nvrhi.h"
#include "RDG/OptiX/OptiXSupport.h"

#define OPTIX_CHECK_LOG(call)                                                                     \
    do                                                                                            \
    {                                                                                             \
        OptixResult res = call;                                                                   \
        const size_t sizeof_log_returned = sizeof_log;                                            \
        sizeof_log = sizeof(optix_log); /* reset sizeof_log for future calls */                   \
        if (res != OPTIX_SUCCESS)                                                                 \
        {                                                                                         \
            std::stringstream ss;                                                                 \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__ << ")\nLog:\n" \
               << optix_log << (sizeof_log_returned > sizeof(optix_log) ? "<TRUNCATED>" : "")     \
               << "\n";                                                                           \
            printf("%s", ss.str().c_str());                                                       \
        }                                                                                         \
    } while (0)

#define OPTIX_CHECK(call)                                                                    \
    do                                                                                       \
    {                                                                                        \
        OptixResult res = call;                                                              \
        if (res != OPTIX_SUCCESS)                                                            \
        {                                                                                    \
            std::stringstream ss;                                                            \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__ << ")\n"; \
            printf("%s", ss.str().c_str());                                                  \
        }                                                                                    \
    } while (0)

char optix_log[2048];

namespace nvrhi
{
    detail::OptiXModule::OptiXModule(const OptiXModuleDesc& desc, IDevice* device)
        : desc(desc)
    {
        size_t sizeof_log = sizeof(optix_log);

        if (!desc.ptx.empty())
        {
            OPTIX_CHECK_LOG(
                optixModuleCreate(
                    device->OptixContext(),
                    &desc.module_compile_options,
                    &desc.pipeline_compile_options,
                    desc.ptx.c_str(),
                    desc.ptx.size(),
                    optix_log,
                    &sizeof_log,
                    &module));
        }
        else
        {
            OPTIX_CHECK(
                optixBuiltinISModuleGet(
                    device->OptixContext(),
                    &desc.module_compile_options,
                    &desc.pipeline_compile_options,
                    &desc.builtinISOptions,
                    &module));
        }
    }

    detail::OptiXProgramGroup::OptiXProgramGroup(
        OptiXProgramGroupDesc desc,
        OptiXModuleHandle module,
        IDevice* device)
        : desc(desc)
    {
        desc.prog_group_desc.raygen.module = module->getModule();

        if (desc.prog_group_desc.kind == OPTIX_PROGRAM_GROUP_KIND_CALLABLES)
        {
            desc.prog_group_desc.callables.moduleDC = nullptr;
            desc.prog_group_desc.callables.entryFunctionNameDC = nullptr;
            desc.prog_group_desc.callables.moduleCC = module->getModule();
        }

        size_t sizeof_log = sizeof(optix_log);
        OPTIX_CHECK_LOG(
            optixProgramGroupCreate(
                device->OptixContext(),
                &desc.prog_group_desc,
                1, // num program groups
                &desc.program_group_options,
                optix_log,
                &sizeof_log,
                &hitgroup_prog_group));
    }

    detail::OptiXProgramGroup::OptiXProgramGroup(
        OptiXProgramGroupDesc desc,
        std::tuple<OptiXModuleHandle, OptiXModuleHandle, OptiXModuleHandle> modules,
        IDevice* device)
        : desc(desc)
    {
        assert(desc.prog_group_desc.kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP);

        if (std::get<0>(modules))
            desc.prog_group_desc.hitgroup.moduleIS = std::get<0>(modules)->getModule();
        else
            desc.prog_group_desc.hitgroup.moduleIS = nullptr;

        if (std::get<1>(modules))
            desc.prog_group_desc.hitgroup.moduleAH = std::get<1>(modules)->getModule();
        else
            desc.prog_group_desc.hitgroup.moduleAH = nullptr;

        if (std::get<2>(modules))
            desc.prog_group_desc.hitgroup.moduleCH = std::get<2>(modules)->getModule();
        else
            throw std::runtime_error("A Closest hit shader must be specified.");

        size_t sizeof_log = sizeof(optix_log);
        OPTIX_CHECK_LOG(
            optixProgramGroupCreate(
                device->OptixContext(),
                &desc.prog_group_desc,
                1, // num program groups
                &desc.program_group_options,
                optix_log,
                &sizeof_log,
                &hitgroup_prog_group));
    }

    detail::OptiXPipeline::OptiXPipeline(
        OptiXPipelineDesc desc,
        const std::vector<OptiXProgramGroupHandle>& program_groups,
        IDevice* device)
        : desc(desc)
    {
        size_t sizeof_log = sizeof(optix_log);

        std::vector<OptixProgramGroup> concrete_program_groups;
        for (int i = 0; i < program_groups.size(); ++i)
        {
            concrete_program_groups.push_back(program_groups[i]->getProgramGroup());
        }

        OPTIX_CHECK_LOG(
            optixPipelineCreate(
                device->OptixContext(),

                &desc.pipeline_compile_options,
                &desc.pipeline_link_options,
                concrete_program_groups.data(),
                program_groups.size(),
                optix_log,
                &sizeof_log,
                &pipeline));

        OptixStackSizes stack_sizes = {};

        for (auto& prog_group : concrete_program_groups)
        {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;

        const uint32_t max_trace_depth = desc.pipeline_link_options.maxTraceDepth;

        OPTIX_CHECK(
            optixUtilComputeStackSizes(
                &stack_sizes,
                max_trace_depth+1,
                max_trace_depth+1, // maxCCDepth
                0,                                 // maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size));
        OPTIX_CHECK(
            optixPipelineSetStackSize(
                pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_trace_depth+2 // maxTraversableDepth
            ));
    }

    detail::OptiXTraversable::OptiXTraversable(const OptiXTraversableDesc& desc, IDevice* device)
        : desc(desc)
    {
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(
            optixAccelComputeMemoryUsage(
                device->OptixContext(), &desc.buildOptions, &desc.buildInput, 1, &gas_buffer_sizes))
        ;

        CUdeviceptr deviceTempBufferGAS;

        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(&deviceTempBufferGAS), gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(&traversableBuffer), gas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(
            optixAccelBuild(
                device->OptixContext(),
                0,
                // CUDA stream
                &desc.buildOptions,
                &desc.buildInput,
                1,
                // num build inputs
                deviceTempBufferGAS,
                gas_buffer_sizes.tempSizeInBytes,
                traversableBuffer,
                gas_buffer_sizes.outputSizeInBytes,
                &handle,
                nullptr,
                // emitted property list
                0 // num emitted properties
            ));
        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaFree((void*)deviceTempBufferGAS));
        CUDA_SYNC_CHECK();
    }

    detail::OptiXTraversable::~OptiXTraversable()
    {
        cudaFree((void*)traversableBuffer);
    }

    static void
    context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
    {
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
            << "]: " << message << "\n";
    }

    void IDevice::OptixPrepare()
    {
        if (!isOptiXInitalized)
        {
            // Initialize CUDA
            CUDA_CHECK(cudaFree(0));

            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(0, &options, &optixContext));
            CUDA_CHECK(cudaStreamCreate(&optixStream));
        }
        isOptiXInitalized = true;
    }
} // namespace nvrhi

#endif
