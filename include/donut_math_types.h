#pragma once

using uint = unsigned;

#ifdef __CUDACC__
using float3x4 = glm::mat3x4;
using float4x4 = glm::mat4x4;
#else
#include "donut/core/math/math.h"
using donut::math::float3x4;
using donut::math::float4x4;
#endif
