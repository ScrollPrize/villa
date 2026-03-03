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
