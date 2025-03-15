cmake_minimum_required(VERSION 3.5)
project(sheet_generation)

# Set the build type if it's not already set.
# set(CMAKE_BUILD_TYPE Debug)
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
# Build HDF5 (version 1.12.2) via ExternalProject_Add.
# ---------------------------
include(ExternalProject)
ExternalProject_Add(hdf5_project
    URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.gz
    PREFIX ${CMAKE_BINARY_DIR}/hdf5
    CMAKE_ARGS 
       -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/hdf5_install
       -DBUILD_SHARED_LIBS=ON
       -DHDF5_ENABLE_THREADSAFE=ON
       -DHDF5_USE_FILE_LOCKING=FALSE
       -DHDF5_BUILD_HL=OFF
       -DHDF5_BUILD_CPP_LIB=OFF
       -DHDF5_BUILD_STATIC_LIBS=OFF
       -DHDF5_BUILD_TOOLS=OFF
       -DHDF5_BUILD_EXAMPLES=OFF
       -DHDF5_BUILD_TESTING=OFF
       -DALLOW_UNSUPPORTED=ON
    UPDATE_DISCONNECTED 1
)

# Create an imported target for HDF5 using the installed files.
add_library(HDF5::HDF5 UNKNOWN IMPORTED)
set_target_properties(HDF5::HDF5 PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/hdf5_install/lib/libhdf5.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/hdf5_install/include"
)

# ---------------------------
# Place the pybind11 modules in the top-level build directory.
# ---------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------
# Create pybind11 modules.
# ---------------------------
pybind11_add_module(sheet_generation solver.cpp)
pybind11_add_module(pointcloud_processing pointcloud_processing.cpp)
pybind11_add_module(meshing_utils meshing_utils.cpp)

# ---------------------------
# Link libraries for each module.
# ---------------------------
target_link_libraries(sheet_generation PRIVATE 
    yaml-cpp 
    Eigen3::Eigen 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${PYTHON_LIBRARIES}
)

target_link_libraries(pointcloud_processing PRIVATE 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${LibArchive_LIBRARIES} 
    TBB::tbb 
    hdbscan 
    OpenMP::OpenMP_CXX 
    ${PYTHON_LIBRARIES} 
    HDF5::HDF5
)

target_link_libraries(meshing_utils PRIVATE 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${LibArchive_LIBRARIES} 
    TBB::tbb 
    hdbscan 
    OpenMP::OpenMP_CXX 
    ${PYTHON_LIBRARIES}
)

# Define compile definitions for thread safety.
target_compile_definitions(pointcloud_processing PRIVATE H5_HAVE_THREADSAFE HDF5_USE_FILE_LOCKING=FALSE)

# Ensure targets that depend on HDF5 wait for its build/install.
add_dependencies(pointcloud_processing hdf5_project)
add_dependencies(meshing_utils hdf5_project)
