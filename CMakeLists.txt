cmake_minimum_required(VERSION 3.19)


file(GLOB_RECURSE rdg_sources src/*.cpp src/*.h)
add_library(RDG ${rdg_sources})

if(RDG_WITH_CUDA)
	if(RDG_WITH_OPTIX)
		set(NVRHI_WITH_CUDA_SUPPORT on)
		set(NVRHI_WITH_OPTIX_SUPPORT on)
	endif()
endif()

set(donut_libs donut_app donut_render)
target_link_libraries(
  RDG
PUBLIC
  ${donut_libs}
)

if(RDG_WITH_CUDA)
	message("Build RDG with CUDA.")
	target_compile_definitions(RDG PUBLIC RDG_WITH_CUDA)
	if(RDG_WITH_OPTIX)
		include(NVRTC.cmake)
		message("Build RDG with OPTIX.")
		target_compile_definitions(RDG PUBLIC RDG_WITH_OPTIX)
	endif()
endif()

target_include_directories(RDG PUBLIC include)
target_compile_features(RDG PUBLIC cxx_std_20)