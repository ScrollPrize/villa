if(NOT DEFINED BUILD_DIR)
    message(FATAL_ERROR "BUILD_DIR is required")
endif()
if(NOT DEFINED CTEST_COMMAND)
    message(FATAL_ERROR "CTEST_COMMAND is required")
endif()
if(NOT DEFINED TEST_TARGETS)
    message(FATAL_ERROR "TEST_TARGETS is required")
endif()
string(REPLACE "," ";" TEST_TARGETS "${TEST_TARGETS}")

set(raw_dir "${BUILD_DIR}/coverage-raw")
set(report_dir "${BUILD_DIR}/coverage")

file(REMOVE_RECURSE "${raw_dir}" "${report_dir}")
file(MAKE_DIRECTORY "${raw_dir}" "${report_dir}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
            "LLVM_PROFILE_FILE=${raw_dir}/%p-%m.profraw"
            "${CTEST_COMMAND}" --output-on-failure
    WORKING_DIRECTORY "${BUILD_DIR}"
    RESULT_VARIABLE ctest_result)

include("${CMAKE_CURRENT_LIST_DIR}/coverage_report.cmake")

if(NOT ctest_result EQUAL 0)
    message(FATAL_ERROR "CTest failed with exit code ${ctest_result}; coverage report was still generated.")
endif()
