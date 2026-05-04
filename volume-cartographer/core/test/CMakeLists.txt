include(FetchContent)

FetchContent_Declare(
    doctest
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    URL https://github.com/doctest/doctest/archive/v2.4.11.tar.gz
)
FetchContent_MakeAvailable(doctest)

add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE doctest::doctest)
add_test(NAME test_smoke COMMAND test_smoke)
