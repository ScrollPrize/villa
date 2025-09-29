# FindPaStiX.cmake
# Exports:
#   PASTIX_FOUND
#   PaStiX::PaStiX (IMPORTED target)
#   PASTIX_INCLUDE_DIR, PASTIX_LIBRARY, PASTIX_VERSION (best effort)
#
# Hints:
#   - Set PASTIX_ROOT or add its prefix to CMAKE_PREFIX_PATH
#

find_path(PASTIX_INCLUDE_DIR
  NAMES pastix.h pastix/pastix.h
  HINTS ${PASTIX_ROOT} ENV PASTIX_ROOT
  PATH_SUFFIXES include
)

find_library(PASTIX_LIBRARY
  NAMES pastix
  HINTS ${PASTIX_ROOT} ENV PASTIX_ROOT
  PATH_SUFFIXES lib lib64
)

# Try pkg-config if available (optional)
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND AND NOT PASTIX_INCLUDE_DIR)
  pkg_check_modules(PC_PASTIX QUIET pastix)
  if(PC_PASTIX_FOUND)
    set(PASTIX_INCLUDE_DIR ${PC_PASTIX_INCLUDEDIR})
    if(NOT PASTIX_LIBRARY AND PC_PASTIX_LINK_LIBRARIES)
      list(GET PC_PASTIX_LINK_LIBRARIES 0 PASTIX_LIBRARY)
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PaStiX
  REQUIRED_VARS PASTIX_INCLUDE_DIR PASTIX_LIBRARY
  FAIL_MESSAGE "PaStiX not found. Set PASTIX_ROOT or add its prefix to CMAKE_PREFIX_PATH."
)

if(PASTIX_FOUND)
  add_library(PaStiX::PaStiX UNKNOWN IMPORTED)
  set_target_properties(PaStiX::PaStiX PROPERTIES
    IMPORTED_LOCATION "${PASTIX_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${PASTIX_INCLUDE_DIR}"
  )

  # Optional transitive libs (mostly no-ops for shared PaStiX)
  # We do soft discovery to avoid hard failures if not present.
  find_library(SCOTCH_LIBRARY NAMES scotch)
  find_library(SCOTCHERR_LIBRARY NAMES scotcherr)
  find_package(BLAS QUIET)
  find_package(LAPACK QUIET)
  find_library(HWLOC_LIBRARY NAMES hwloc)
  find_library(M_LIB NAMES m)
  find_library(PTHREAD_LIB NAMES pthread)
  find_library(DL_LIB NAMES dl)

  set(_extra_libs "")
  foreach(lib "${HWLOC_LIBRARY}" "${SCOTCH_LIBRARY}" "${SCOTCHERR_LIBRARY}"
               "${BLAS_LIBRARIES}" "${LAPACK_LIBRARIES}"
               "${M_LIB}" "${PTHREAD_LIB}" "${DL_LIB}")
    if(lib)
      list(APPEND _extra_libs ${lib})
    endif()
  endforeach()
  if(_extra_libs)
    target_link_libraries(PaStiX::PaStiX INTERFACE ${_extra_libs})
  endif()
endif()

mark_as_advanced(PASTIX_INCLUDE_DIR PASTIX_LIBRARY)