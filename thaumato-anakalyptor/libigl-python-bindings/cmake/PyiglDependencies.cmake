# Prepare dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# this project.

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

################################################################################
# Required libraries
################################################################################

include(FetchContent)

# Define path to local libigl
set(LOCAL_LIBIGL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/libigl")

# Check if libigl already exists in the local directory
if(EXISTS "${LOCAL_LIBIGL_DIR}/CMakeLists.txt")
    message(STATUS "Using local libigl found at ${LOCAL_LIBIGL_DIR}")
    add_subdirectory(${LOCAL_LIBIGL_DIR} libigl)

    # Define the necessary target for later use
    set(libigl_SOURCE_DIR ${LOCAL_LIBIGL_DIR})
    set(LIBIGL_FOUND TRUE)
else()
    message(STATUS "Local libigl not found, fetching from GitHub...")

    FetchContent_Declare(
        libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG f962e4a6b68afe978dc12a63702b7846a3e7a6ed
    )
    FetchContent_GetProperties(libigl)

    if(NOT libigl_POPULATED)
        FetchContent_MakeAvailable(libigl)
    endif()

    set(LIBIGL_FOUND TRUE)
endif()

# Ensure that libigl is properly included before checking for dependencies
if(LIBIGL_FOUND)
    message(STATUS "libigl is available, checking dependencies...")
    
    # Check if CGAL is required
    if(LIBIGL_COPYLEFT_CGAL)
        find_package(CGAL REQUIRED COMPONENTS Core)
        if(CGAL_FOUND)
            message(STATUS "CGAL found: ${CGAL_INCLUDE_DIRS}")
        else()
            message(FATAL_ERROR "CGAL not found. Install CGAL and try again.")
        endif()
    endif()
endif()

# Fetch NumpyEigen
FetchContent_Declare(
    numpyeigen
    GIT_REPOSITORY https://github.com/fwilliams/numpyeigen.git
    GIT_TAG 14acc7a71285979016ef39041d8cd4df97e4e829
)

# Check if population has already been performed
FetchContent_GetProperties(numpyeigen)
if(NOT numpyeigen_POPULATED)
    FetchContent_Populate(numpyeigen)
endif()

# Push CMAKE_MODULE_PATH
set(PREV_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${numpyeigen_SOURCE_DIR}/cmake)

# Define Eigen path
SET(NPE_WITH_EIGEN "${libigl_SOURCE_DIR}/../eigen-src/" CACHE INTERNAL "")

include(numpyeigen)

# Pop CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${PREV_CMAKE_MODULE_PATH})
