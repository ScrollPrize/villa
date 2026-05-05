include(FetchContent)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "" FORCE)

FetchContent_Declare(
    doctest
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    URL https://github.com/doctest/doctest/archive/v2.4.12.tar.gz
)
FetchContent_MakeAvailable(doctest)

add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE doctest::doctest)
add_test(NAME test_smoke COMMAND test_smoke)

add_executable(test_appendmask_render test_appendmask_render.cpp)
target_link_libraries(test_appendmask_render PRIVATE doctest::doctest vc_core)
add_test(NAME test_appendmask_render COMMAND test_appendmask_render)
