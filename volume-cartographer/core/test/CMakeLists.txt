function(vc_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE vc_core)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

vc_add_test(test_version test_version.cpp)
vc_add_test(test_geometry test_geometry.cpp)
