cmake_minimum_required(VERSION 3.19)

file(GLOB_RECURSE rdg_sources src/*.cpp src/*.h)

if(RDG_WITH_CUDA)
	file(GLOB_RECURSE rdg_sources src/*.cpp src/*.h src/*.cu)
	if(RDG_WITH_OPTIX)
		set(NVRHI_WITH_CUDA_SUPPORT on)
		set(NVRHI_WITH_OPTIX_SUPPORT on)
	endif()
endif()

add_library(RDG ${rdg_sources})
if(RDG_WITH_CUDA)
	set_target_properties(RDG
	    PROPERTIES 
	    CUDA_RESOLVE_DEVICE_SYMBOLS ON
	    CUDA_SEPARABLE_COMPILATION ON
	)
    target_compile_options(RDG
        PUBLIC
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;--forward-unknown-to-host-compiler;-g;-lineinfo;-rdc=true;-c;>"
    )
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