# utils — header-only C++23 utility library (vendored)
# Source: https://github.com/SuperOptimizer/utils

add_library(utils INTERFACE)
target_include_directories(utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(utils INTERFACE cxx_std_23)

# Individual header targets (for granular dependency management)
set(UTILS_HEADERS
    hash
    lru_cache
    lock_pool
    priority_queue
)

foreach(header ${UTILS_HEADERS})
    add_library(utils_${header} INTERFACE)
    target_include_directories(utils_${header} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_features(utils_${header} INTERFACE cxx_std_23)
endforeach()

# Tests
if(VC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
