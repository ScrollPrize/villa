find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(test_verified_ct_read verified_ct_read.cpp)
target_link_libraries(test_verified_ct_read PRIVATE utils)
set_target_properties(test_verified_ct_read PROPERTIES
    OUTPUT_NAME integrated_reader_fixture
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/verified-ct-read")
vc_register_test_target(test_verified_ct_read)
add_test(NAME verified_ct_read
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/verified_ct_read.py
            ${CMAKE_CURRENT_BINARY_DIR}/verified-ct-read)
vc_register_ctest(verified_ct_read test_verified_ct_read)
