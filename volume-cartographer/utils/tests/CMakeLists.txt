# Helper function to add a test for a specific header
function(add_utils_test name)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.cpp")
        message(STATUS "Skipping test_${name} (source file not found)")
        return()
    endif()
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE utils)
    target_compile_features(test_${name} PRIVATE cxx_std_23)
    add_test(NAME ${name} COMMAND test_${name})
endfunction()

# Foundation
add_utils_test(hash)

# Data Structures
add_utils_test(lru_cache)
add_utils_test(lock_pool)
add_utils_test(priority_queue)

# Algorithms
add_utils_test(compositing)

# Zarr + compression (need compression libs)
find_library(ZSTD_LIB zstd)
find_library(LZ4_LIB lz4)
find_library(ZLIB_LIB z)
find_library(BLOSC_LIB blosc)

add_utils_test(zarr)
if(TARGET test_zarr)
    if(ZSTD_LIB)
        target_link_libraries(test_zarr PRIVATE ${ZSTD_LIB})
    endif()
    if(LZ4_LIB)
        target_link_libraries(test_zarr PRIVATE ${LZ4_LIB})
    endif()
    if(ZLIB_LIB)
        target_link_libraries(test_zarr PRIVATE ${ZLIB_LIB})
    endif()
    if(BLOSC_LIB)
        target_link_libraries(test_zarr PRIVATE ${BLOSC_LIB})
    endif()
endif()

# I/O & Systems
add_utils_test(thread_pool)

# Compression (need compression libs)
add_utils_test(compression)
if(TARGET test_compression)
    if(ZSTD_LIB)
        target_link_libraries(test_compression PRIVATE ${ZSTD_LIB})
    endif()
    if(LZ4_LIB)
        target_link_libraries(test_compression PRIVATE ${LZ4_LIB})
    endif()
    if(ZLIB_LIB)
        target_link_libraries(test_compression PRIVATE ${ZLIB_LIB})
    endif()
    if(BLOSC_LIB)
        target_link_libraries(test_compression PRIVATE ${BLOSC_LIB})
    endif()
endif()
