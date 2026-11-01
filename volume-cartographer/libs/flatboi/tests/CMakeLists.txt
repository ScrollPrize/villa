add_executable(test_slim_assembly test_slim_assembly.cpp)
set_target_properties(test_slim_assembly PROPERTIES
    AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
target_compile_definitions(test_slim_assembly PRIVATE SLIM_CACHED PASTIX6)
target_include_directories(test_slim_assembly PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../core/test")
target_link_libraries(test_slim_assembly PRIVATE
    igl::core pastix spm OpenMP::OpenMP_CXX)

# This directory is added before the top-level instrumentation flags, so apply
# them explicitly to the test without instrumenting the PaStiX dependencies.
if(_san)
    target_compile_options(test_slim_assembly PRIVATE ${_san})
    target_link_options(test_slim_assembly PRIVATE ${_san})
endif()
if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN OR VC_ENABLE_LSAN
   OR VC_ENABLE_TYSAN OR VC_ENABLE_NSAN)
    target_sources(test_slim_assembly PRIVATE
        "${CMAKE_SOURCE_DIR}/misc/sanitizer_config.cpp")
endif()

set(_slim_test_environment
    PASTIX_NUM_THREADS=2 OMP_NUM_THREADS=2
    OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2)
add_custom_target(check_slim_assembly
    COMMAND ${CMAKE_COMMAND} -E env ${_slim_test_environment}
            $<TARGET_FILE:test_slim_assembly>
    DEPENDS test_slim_assembly
    USES_TERMINAL)

if(VC_TESTING)
    add_test(NAME test_slim_assembly COMMAND test_slim_assembly)
    set_tests_properties(test_slim_assembly PROPERTIES
        LABELS "vc-flatboi"
        ENVIRONMENT "${_slim_test_environment}"
        PROCESSORS 2
        TIMEOUT 120)
endif()
