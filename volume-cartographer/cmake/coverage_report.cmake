if(NOT DEFINED BUILD_DIR)
    message(FATAL_ERROR "BUILD_DIR is required")
endif()
if(NOT DEFINED LLVM_PROFDATA)
    message(FATAL_ERROR "LLVM_PROFDATA is required")
endif()
if(NOT DEFINED LLVM_COV)
    message(FATAL_ERROR "LLVM_COV is required")
endif()

set(profdata "${BUILD_DIR}/coverage.profdata")
set(report_dir "${BUILD_DIR}/coverage")
set(raw_dir "${BUILD_DIR}/coverage-raw")
set(ignore_regex "(/_deps/|/build/|/libs/)")

file(GLOB profraw_files LIST_DIRECTORIES false "${raw_dir}/*.profraw")
if(NOT profraw_files)
    message(FATAL_ERROR "No .profraw files found in ${raw_dir}")
endif()

execute_process(
    COMMAND "${LLVM_PROFDATA}" merge -sparse -o "${profdata}" ${profraw_files}
    COMMAND_ERROR_IS_FATAL ANY)

set(test_bins)
foreach(test_target IN LISTS TEST_TARGETS)
    list(APPEND test_bins "${BUILD_DIR}/bin/${test_target}")
endforeach()
if(NOT test_bins)
    message(FATAL_ERROR "No test binaries found in ${BUILD_DIR}/bin")
endif()

set(object_args)
foreach(test_bin IN LISTS test_bins)
    list(APPEND object_args -object "${test_bin}")
endforeach()

execute_process(
    COMMAND "${LLVM_COV}" show ${object_args}
            "-instr-profile=${profdata}"
            -format=html
            "-output-dir=${report_dir}"
            "-ignore-filename-regex=${ignore_regex}"
            -show-line-counts-or-regions
    COMMAND_ERROR_IS_FATAL ANY)

execute_process(
    COMMAND "${LLVM_COV}" export ${object_args}
            "-instr-profile=${profdata}"
            -format=lcov
            "-ignore-filename-regex=${ignore_regex}"
    OUTPUT_FILE "${report_dir}/lcov.info"
    COMMAND_ERROR_IS_FATAL ANY)

execute_process(
    COMMAND "${LLVM_COV}" report ${object_args}
            "-instr-profile=${profdata}"
            "-ignore-filename-regex=${ignore_regex}"
    OUTPUT_FILE "${report_dir}/summary.txt"
    COMMAND_ERROR_IS_FATAL ANY)

message(STATUS "Coverage HTML: ${report_dir}/index.html")
message(STATUS "Coverage summary: ${report_dir}/summary.txt")
