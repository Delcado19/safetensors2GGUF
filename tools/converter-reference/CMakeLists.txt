cmake_minimum_required(VERSION 3.14...3.28)
project(converter_reference C CXX)

if(NOT DEFINED REFERENCE_SOURCE)
    message(FATAL_ERROR "REFERENCE_SOURCE must point to a verified frozen llama.cpp tree")
endif()

# Reference math only: avoid host-specific SIMD and optional SDK/network backends.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
foreach(option NATIVE AVX AVX2 AVX512 FMA F16C OPENMP CUDA METAL BLAS VULKAN KOMPUTE CCACHE)
    set(GGML_${option} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory("${REFERENCE_SOURCE}/ggml" ggml)
add_executable(converter-reference probe.c)
target_include_directories(converter-reference PRIVATE "${REFERENCE_SOURCE}/ggml/src")
target_link_libraries(converter-reference PRIVATE ggml)
if(NOT MSVC)
    target_link_libraries(converter-reference PRIVATE m)
endif()
