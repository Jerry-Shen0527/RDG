#pragma once

#define OptiX_DIR "C:/Users/Jerry/Workspace/Furnace"
#define OptiX_PTX_DIR "C:/Users/Jerry/Workspace/Furnace/lib/ptx"
#define OptiX_CUDA_DIR "C:/Users/Jerry/Workspace/Furnace/src/OptiXShaders"

// Include directories
#define OptiX_RELATIVE_INCLUDE_DIRS \
  "include", \
  ".",  "C:/Users/Jerry/Workspace/Furnace/include",
#define OptiX_ABSOLUTE_INCLUDE_DIRS \
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0/include", \
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include/cuda/std", \
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include",  "C:/Users/Jerry/Workspace/Furnace/ext/glm/"

// Signal whether to use NVRTC or not
#define CUDA_NVRTC_ENABLED 1

// NVRTC compiler options
#define CUDA_NVRTC_OPTIONS  \
  "-std=c++17", \
  "-arch", \
  "compute_86", \
  "-lineinfo", \
  "-use_fast_math", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64",
