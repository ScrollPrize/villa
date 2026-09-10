##====================================================
#                  Find VCG Library
#-----------------------------------------------------

find_path(VCG_DIR "vcg/complex/complex.h"
    HINTS
        "${VCG_ROOT}"
        "$ENV{VCG_ROOT}"
    PATHS
        "$ENV{PROGRAMFILES}"
        "$ENV{PROGRAMW6432}"
        "/usr"
        "/usr/local"
        "/usr/share"
        "/usr/local/share"
        "/usr/lib/x86_64-linux-gnu/cmake"
    PATH_SUFFIXES
        "vcg"
        "include"
    DOC
        "Root directory of VCG library")

##====================================================
## Include VCG library
##----------------------------------------------------
if(EXISTS ${VCG_DIR})
    set(VCG_FOUND TRUE)
    set(VCG_INCLUDE_DIRS ${VCG_DIR})
    set(VCG_DIR "${VCG_DIR}" CACHE PATH "" FORCE)
    mark_as_advanced(VCG_DIR)
    set(VCG_INCLUDE_DIR ${VCG_DIR})

    message(STATUS "VCG ${VCG_VERSION} found (include: ${VCG_INCLUDE_DIRS})")
else()
    if(VCG_FIND_REQUIRED)
        message(FATAL_ERROR "VCG required but not found.")
    endif(VCG_FIND_REQUIRED)
endif()
##====================================================