# FindJsonC.cmake
# Defines:
#  JSONC_FOUND        - System has json-c
#  JSONC_INCLUDE_DIRS - json-c include directories
#  JSONC_LIBRARY      - json-c library
#  JsonC::JsonC       - Imported target

find_path(JSONC_INCLUDE_DIRS
        NAMES json-c/json.h
        PATHS ${JSONC_ROOT_DIR}
        PATH_SUFFIXES include
)

find_library(JSONC_LIBRARY
        NAMES json-c
        PATHS ${JSONC_ROOT_DIR}
        PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JsonC
        REQUIRED_VARS JSONC_LIBRARY JSONC_INCLUDE_DIRS
)

if(JsonC_FOUND AND NOT TARGET JsonC::JsonC)
    add_library(JsonC::JsonC UNKNOWN IMPORTED)
    set_target_properties(JsonC::JsonC PROPERTIES
            IMPORTED_LOCATION "${JSONC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${JSONC_INCLUDE_DIRS}"
    )
endif()

mark_as_advanced(JSONC_INCLUDE_DIRS JSONC_LIBRARY)