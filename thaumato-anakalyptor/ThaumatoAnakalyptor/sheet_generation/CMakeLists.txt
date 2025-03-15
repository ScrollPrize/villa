cmake_minimum_required(VERSION 3.5)
project(sheet_generation)

# Set build type, C++ standard, and optimization flags.
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})

# Add subdirectories for dependencies.
add_subdirectory(pybind11)
add_subdirectory(yaml-cpp)
add_subdirectory(hdbscan)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)        # Threading library
find_package(LibArchive REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

# ---------------------------
# Fetch and build HDF5 (version 1.12.2) with thread safety enabled
# and disable the unsupported C++ API.
# ---------------------------
include(FetchContent)

# Allow unsupported options.
set(ALLOW_UNSUPPORTED ON CACHE BOOL "Allow unsupported options")

FetchContent_Declare(
  hdf5
  URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.gz
)
# Set options for a thread-safe build.
set(HDF5_ENABLE_THREADSAFE ON CACHE BOOL "Enable HDF5 thread safety")
set(HDF5_USE_FILE_LOCKING FALSE CACHE BOOL "Disable HDF5 file locking")
set(HDF5_BUILD_HL OFF CACHE BOOL "Disable HDF5 HL API")
set(HDF5_BUILD_CPP_LIB OFF CACHE BOOL "Disable HDF5 C++ API")
set(HDF5_BUILD_TOOLS OFF CACHE BOOL "Disable HDF5 tools")
set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "Disable HDF5 examples")
set(HDF5_BUILD_TESTING OFF CACHE BOOL "Disable HDF5 testing")
# Specify an install prefix for HDF5.
set(HDF5_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/hdf5_install CACHE PATH "HDF5 install prefix")

FetchContent_MakeAvailable(hdf5)

# Create the install directories so that the imported target’s properties reference existing paths.
file(MAKE_DIRECTORY "${HDF5_INSTALL_PREFIX}/include")
file(MAKE_DIRECTORY "${HDF5_INSTALL_PREFIX}/lib")

# ---------------------------
# Add a custom target to install HDF5.
# ---------------------------
add_custom_target(hdf5_install
  COMMAND ${CMAKE_COMMAND} --install ${hdf5_BINARY_DIR} --prefix ${HDF5_INSTALL_PREFIX}
  COMMENT "Installing HDF5 to ${HDF5_INSTALL_PREFIX}"
)

# Create an imported target for HDF5 if one is not provided.
if (NOT TARGET HDF5::HDF5)
  add_library(HDF5::HDF5 UNKNOWN IMPORTED)
  set_target_properties(HDF5::HDF5 PROPERTIES
    IMPORTED_LOCATION "${HDF5_INSTALL_PREFIX}/lib/libhdf5.so"
    INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INSTALL_PREFIX}/include")
endif()

# ---------------------------
# Create pybind11 modules.
# ---------------------------
pybind11_add_module(sheet_generation solver.cpp)
pybind11_add_module(pointcloud_processing pointcloud_processing.cpp)
pybind11_add_module(meshing_utils meshing_utils.cpp)

# Link libraries for each module.
target_link_libraries(sheet_generation PRIVATE yaml-cpp Eigen3::Eigen ${CMAKE_THREAD_LIBS_INIT} ${PYTHON_LIBRARIES})

target_link_libraries(pointcloud_processing PRIVATE 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${LibArchive_LIBRARIES} 
    TBB::tbb 
    hdbscan 
    OpenMP::OpenMP_CXX 
    ${PYTHON_LIBRARIES} 
    HDF5::HDF5)

target_link_libraries(meshing_utils PRIVATE 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${LibArchive_LIBRARIES} 
    TBB::tbb 
    hdbscan 
    OpenMP::OpenMP_CXX 
    ${PYTHON_LIBRARIES})

# Define compile definitions so that our code and HDF5 headers know we use thread safety.
target_compile_definitions(pointcloud_processing PRIVATE H5_HAVE_THREADSAFE HDF5_USE_FILE_LOCKING=FALSE)

# Ensure that targets depending on HDF5 are built after hdf5_install.
add_dependencies(pointcloud_processing hdf5_install)
add_dependencies(meshing_utils hdf5_install)
