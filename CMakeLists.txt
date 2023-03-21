cmake_minimum_required(VERSION 3.19)

file(GLOB_RECURSE rdg_sources src/*.cpp src/*.h)

add_library(RDG ${rdg_sources})

set(donut_libs donut_app donut_render)
target_link_libraries(
  RDG
PUBLIC
  ${donut_libs}
)
target_include_directories(RDG PUBLIC include)
target_compile_features(RDG PUBLIC cxx_std_20)