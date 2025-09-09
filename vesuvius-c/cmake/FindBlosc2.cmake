# FindBlosc2.cmake
# Defines:
#  BLOSC2_FOUND        - System has Blosc2
#  BLOSC2_INCLUDE_DIRS - Blosc2 include directories
#  BLOSC2_LIBRARY      - Blosc2 library
#  Blosc2::Blosc2      - Imported target

find_path(BLOSC2_INCLUDE_DIRS
        NAMES blosc2.h
        PATHS ${BLOSC2_ROOT_DIR}
        PATH_SUFFIXES include
)

find_library(BLOSC2_LIBRARY
        NAMES blosc2
        PATHS ${BLOSC2_ROOT_DIR}
        PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Blosc2
        REQUIRED_VARS BLOSC2_LIBRARY BLOSC2_INCLUDE_DIRS
)

if(Blosc2_FOUND AND NOT TARGET Blosc2::Blosc2)
    add_library(Blosc2::Blosc2 UNKNOWN IMPORTED)
    set_target_properties(Blosc2::Blosc2 PROPERTIES
            IMPORTED_LOCATION "${BLOSC2_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${BLOSC2_INCLUDE_DIRS}"
    )
endif()

mark_as_advanced(BLOSC2_INCLUDE_DIRS BLOSC2_LIBRARY)