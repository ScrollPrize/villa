# Feature check for the CUDA backend: can the toolkit's nvcc compile what the backend
# uses from CUB and libcu++? It is a compile test, not a version comparison, so it
# holds for any toolkit that provides the features. Needs the CUDA language enabled.
#
# vc_reader_check_cuda_backend(<ok-var> <why-var>)
#   <ok-var>  set to TRUE if the probe compiled, else FALSE
#   <why-var> set to a message naming the nvcc path and version when it did not
function(vc_reader_check_cuda_backend ok_var why_var)
    set(source "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cuda_backend_probe.cu")
    try_compile(_vc_reader_cuda_probe_compiled "${CMAKE_BINARY_DIR}/CMakeFiles/vc_reader_cuda_backend_probe"
        SOURCES "${source}"
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        OUTPUT_VARIABLE output)
    set(compiled "${_vc_reader_cuda_probe_compiled}")
    unset(_vc_reader_cuda_probe_compiled CACHE)
    if(compiled)
        set(${ok_var} TRUE PARENT_SCOPE)
        return()
    endif()
    string(REGEX MATCH "[^\n]*error[^\n]*" detail "${output}")
    if(NOT detail)
        string(LENGTH "${output}" length)
        math(EXPR from "${length} - 400")
        if(from LESS 0)
            set(from 0)
        endif()
        string(SUBSTRING "${output}" ${from} -1 detail)
    endif()
    set(${ok_var} FALSE PARENT_SCOPE)
    string(CONCAT why
        "The CUDA toolkit at ${CMAKE_CUDA_COMPILER} (nvcc ${CMAKE_CUDA_COMPILER_VERSION}) could not compile "
        "a test program that uses what the CUDA backend needs from CUB and libcu++ "
        "(for example cub::DeviceRadixSort::SortKeys with a decomposer). "
        "The CUDA backend needs a newer CUDA toolkit. Compiler output: ${detail}")
    set(${why_var} "${why}" PARENT_SCOPE)
endfunction()
