if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-unknown-warning-option)
else()
    add_compile_options(
        -Wno-psabi
        -Wno-ignored-attributes
        -Wno-narrowing
    )
endif()

if(VC_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(
            -Weverything
            -Wno-padded
            -Wno-unsafe-buffer-usage -Wno-unsafe-buffer-usage-in-libc-call -Wno-unsafe-buffer-usage-in-container
            -Wno-ctad-maybe-unsupported
            -Wno-disabled-macro-expansion
            -Wno-exit-time-destructors -Wno-global-constructors
            -Wno-newline-eof -Wno-extra-semi -Wno-extra-semi-stmt
            -Wno-weak-vtables -Wno-packed
            -Wno-nrvo
            -Wno-missing-prototypes -Wno-missing-variable-declarations
            -Wno-reserved-identifier -Wno-reserved-macro-identifier
            -Wno-covered-switch-default -Wno-switch-enum -Wno-switch-default
            -Wno-source-uses-openmp
            -Wno-old-style-cast
            -Wno-zero-as-null-pointer-constant
            -Wno-format-nonliteral
            -Wno-comment
            -Wno-documentation -Wno-documentation-unknown-command
            -Wno-comma
            -Wno-c++98-compat -Wno-c++98-compat-pedantic
            -Wno-c++98-compat-local-type-template-args -Wno-c++98-compat-unnamed-type-template-args
            -Wno-c++98-compat-extra-semi
            -Wno-c++11-compat -Wno-c++14-compat -Wno-c++17-compat -Wno-c++20-compat
            -Wno-pre-c++14-compat -Wno-pre-c++17-compat
            -Wno-pre-c++20-compat -Wno-pre-c++23-compat -Wno-pre-c++26-compat
            -Wno-nested-anon-types -Wno-gnu-anonymous-struct
        )
        message(STATUS "VC_WARNINGS: -Weverything with curated exclusions [clang]")
    else()
        add_compile_options(
            -Wall -Wextra -pedantic
            -Wattributes -Wcast-align -Wcast-qual -Wchar-subscripts -Wcomment
            -Wconversion -Wdelete-incomplete -Wdelete-non-virtual-dtor
            -Wenum-compare -Wmain -Wmissing-field-initializers -Wmissing-noreturn
            -Wold-style-cast -Woverloaded-virtual -Wpointer-arith
            -Wtautological-compare -Wundef -Wuninitialized -Wunreachable-code
            -Wunused -Wvla -Wunused-parameter
            -Wlogical-op -Wduplicated-cond -Wduplicated-branches
            -Wnull-dereference -Wuseless-cast -Wsuggest-override
        )
        message(STATUS "VC_WARNINGS: -Wall -Wextra -pedantic + curated extras [gcc]")
    endif()
endif()

if(VC_WERROR)
    add_compile_options(-Werror)
    message(STATUS "VC_WERROR: warnings treated as errors")
endif()
