# AMD GPU support: compile vc_reader's CUDA backends with HIP (ROCm).
#
# The .cu sources are reused as written. src/cpp/hip_compat maps the CUDA
# runtime, cuFFT, CUB and libcu++ names they use to HIP, hipFFT, hipCUB and
# rocPRIM.
#
# Warp-level kernels assume 32-lane warps, so only wave32 GPUs (RDNA: gfx10xx,
# gfx11xx, gfx12xx) are accepted.

get_filename_component(VC_READER_HIP_COMPAT_DIR "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/hip_compat" ABSOLUTE)

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()
if(EXISTS "${ROCM_PATH}")
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    if(NOT DEFINED CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX} AND EXISTS "${ROCM_PATH}/llvm/bin/clang++")
        set(CMAKE_HIP_COMPILER "${ROCM_PATH}/llvm/bin/clang++" CACHE FILEPATH "HIP compiler")
    endif()
endif()

# GPU targets: CMAKE_HIP_ARCHITECTURES if given, else the GPUs of this machine.
if(NOT CMAKE_HIP_ARCHITECTURES)
    find_program(VC_READER_ROCM_AGENT_ENUMERATOR rocm_agent_enumerator HINTS "${ROCM_PATH}/bin")
    if(VC_READER_ROCM_AGENT_ENUMERATOR)
        execute_process(COMMAND "${VC_READER_ROCM_AGENT_ENUMERATOR}"
            OUTPUT_VARIABLE _vcr_agents OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(REGEX MATCHALL "gfx[0-9a-f]+" _vcr_agents "${_vcr_agents}")
        list(REMOVE_ITEM _vcr_agents gfx000)
        list(REMOVE_DUPLICATES _vcr_agents)
        if(_vcr_agents)
            set(CMAKE_HIP_ARCHITECTURES "${_vcr_agents}" CACHE STRING "AMD GPU targets" FORCE)
        endif()
    endif()
    if(NOT CMAKE_HIP_ARCHITECTURES)
        message(FATAL_ERROR "VC_READER_ENABLE_HIP: no AMD GPU detected; "
            "set CMAKE_HIP_ARCHITECTURES to your GPU target, for example gfx1200")
    endif()
endif()
foreach(_vcr_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
    if(NOT _vcr_arch MATCHES "^gfx1[0-9]")
        message(FATAL_ERROR "VC_READER_ENABLE_HIP supports wave32 GPUs (gfx10xx, gfx11xx, gfx12xx). "
            "${_vcr_arch} runs 64-lane wavefronts, which vc_reader's warp-level kernels do not handle yet.")
    endif()
endforeach()
message(STATUS "vc_reader: HIP backends for ${CMAKE_HIP_ARCHITECTURES}")

enable_language(HIP)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
find_package(hip REQUIRED CONFIG)
find_package(hipfft REQUIRED CONFIG)
find_package(hipcub REQUIRED CONFIG)

# Export the HIP dependencies under vc_reader names so consumers can also
# discover the NVIDIA toolkit in the same CMake project.
add_library(vc_reader_hip_runtime INTERFACE)
set_target_properties(vc_reader_hip_runtime PROPERTIES EXPORT_NAME hip_runtime)
target_compile_definitions(vc_reader_hip_runtime INTERFACE VC_READER_HIP=1)
target_link_libraries(vc_reader_hip_runtime INTERFACE hip::host)

# CUDA-named shims are implementation headers. An installed consumer can
# include the real NVIDIA headers alongside vc_reader headers.
include_directories(BEFORE "${VC_READER_HIP_COMPAT_DIR}")

add_library(vc_reader_hip_fft INTERFACE)
set_target_properties(vc_reader_hip_fft PROPERTIES EXPORT_NAME hip_fft)
target_link_libraries(vc_reader_hip_fft INTERFACE vc_reader_hip_runtime hip::hipfft)

set(VC_READER_GPU_RUNTIME vc_reader_hip_runtime)
set(VC_READER_GPU_FFT vc_reader_hip_fft)

# Called once the GPU targets exist: compile their .cu sources as HIP and
# translate the per-file nvcc options.
function(vc_reader_hip_finalize)
    foreach(_target IN ITEMS vc_reader_thinning vc_reader_transform_cuda vc_reader_reader_cuda)
        if(NOT TARGET ${_target})
            continue()
        endif()
        get_target_property(_sources ${_target} SOURCES)
        foreach(_source IN LISTS _sources)
            if(NOT _source MATCHES "\\.cu$")
                continue()
            endif()
            get_source_file_property(_options "${_source}" TARGET_DIRECTORY ${_target} COMPILE_OPTIONS)
            set(_hip_options "")
            if(_options)
                foreach(_option IN LISTS _options)
                    if(_option STREQUAL "--fmad=false")
                        # Separate float rounding, as nvcc --fmad=false.
                        list(APPEND _hip_options "-ffp-contract=off")
                    elseif(_option STREQUAL "--expt-relaxed-constexpr" OR _option MATCHES "-Xcompiler")
                        # Clang treats constexpr functions as host and device, and
                        # compiles host code itself, so these nvcc options have no
                        # HIP counterpart.
                    else()
                        list(APPEND _hip_options "${_option}")
                    endif()
                endforeach()
            endif()
            set_source_files_properties("${_source}" TARGET_DIRECTORY ${_target}
                PROPERTIES LANGUAGE HIP COMPILE_OPTIONS "${_hip_options}")
        endforeach()
        set_target_properties(${_target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
        target_compile_options(${_target} PRIVATE
            "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include ${VC_READER_HIP_COMPAT_DIR}/vcr_hip_prelude.h>")
        # hipCUB and rocPRIM are header-only. Their package targets would also add
        # "-x hip" to every C++ source of a consumer, so only their include
        # directories are used, for the HIP sources.
        foreach(_headers IN ITEMS hip::hipcub roc::rocprim)
            if(TARGET ${_headers})
                target_include_directories(${_target} SYSTEM PRIVATE
                    "$<$<COMPILE_LANGUAGE:HIP>:$<TARGET_PROPERTY:${_headers},INTERFACE_INCLUDE_DIRECTORIES>>")
            endif()
        endforeach()
    endforeach()
endfunction()
