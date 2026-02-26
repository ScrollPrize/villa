# VCSanitizers.cmake - Sanitizer and debug instrumentation

if(VC_ENABLE_ASAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be used together")
endif()
if(VC_ENABLE_LSAN AND VC_ENABLE_ASAN)
    message(WARNING "LeakSanitizer is already included with AddressSanitizer. Disabling standalone LSAN.")
    set(VC_ENABLE_LSAN OFF)
endif()
if(VC_ENABLE_LSAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "LeakSanitizer and ThreadSanitizer cannot be used together")
endif()

if(VC_ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -Og -DADDRESS_SANITIZER")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
endif()

if(VC_ENABLE_UBSAN)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
    set(UBSAN_FLAGS "-fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=nullability")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UBSAN_FLAGS} -fno-omit-frame-pointer -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${UBSAN_FLAGS} -fno-omit-frame-pointer -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${UBSAN_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${UBSAN_FLAGS}")
endif()

if(VC_ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread -fno-omit-frame-pointer -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(VC_ENABLE_LSAN)
    message(STATUS "LeakSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak -fno-omit-frame-pointer -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=leak")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=leak")
endif()

# Code coverage (works with both GCC and Clang)
# Usage:
#   cmake -DVC_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug ..
#   cmake --build . --target VC3D
#   ./bin/VC3D                    # or run tests, or use the app normally
#   # GCC: generates .gcda/.gcno files, use gcovr/lcov to report
#   # Clang: generates .profraw, merge with llvm-profdata, report with llvm-cov
#
# Quick report with gcovr (GCC):
#   gcovr -r ../core --html-details coverage.html
#
# Quick report with llvm-cov (Clang):
#   llvm-profdata merge -sparse *.profraw -o merged.profdata
#   llvm-cov show ./bin/VC3D -instr-profile=merged.profdata -format=html > coverage.html
if(VC_ENABLE_COVERAGE)
    message(STATUS "Code coverage enabled")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping -fno-lto -O0 -g")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-generate -fcoverage-mapping -fno-lto -O0 -g")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-instr-generate -fno-lto")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fprofile-instr-generate -fno-lto")
    else()
        # GCC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fno-lto -O0 -g")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fno-lto -O0 -g")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage -fno-lto")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage -fno-lto")
    endif()
endif()

# Valgrind: disable LTO, restrict ISA to baseline x86-64
if(VC_USE_VALGRIND)
    message(STATUS "Valgrind support enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto -march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-avx512f -mno-fma -mno-bmi -mno-bmi2 -Og -g -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-avx512cd -mno-avx512bw -mno-avx512dq -mno-avx512vl")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-avx512f -mno-avx512cd -mno-avx512bw -mno-avx512dq -mno-avx512vl -mno-fma")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-lto")
endif()
