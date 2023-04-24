#pragma once

#ifdef RDG_WITH_OPTIX

#include <optix.h>


#include <string>
#include <vector>


struct PipelineWithSbt
{
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
};

using OptixShaderFunc = std::tuple<std::string, OptixModule>;

/**
 * \brief IS, CHS, AHS
 */
using HitGroup = std::tuple<OptixShaderFunc, OptixShaderFunc, OptixShaderFunc>;

#define GetEntryName(ShaderFunc) std::get<0>(ShaderFunc)
#define GetModule(ShaderFunc)    std::get<1>(ShaderFunc)

#define GetIS(HitGroup)  std::get<0>(HitGroup)
#define GetCHS(HitGroup) std::get<1>(HitGroup)
#define GetAHS(HitGroup) std::get<2>(HitGroup)

extern char optix_log[2048];

#endif
