cmake_minimum_required(VERSION 3.5)
project(sheet_generation)

# Set the build type if it's not already set
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# Add optimization flags for the release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})

# Add pybind11 submodule
add_subdirectory(pybind11)

# Add YAML-CPP submodule
add_subdirectory(yaml-cpp)

# Add hdbscan submodule
add_subdirectory(hdbscan)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)        # Find the threading library
find_package(LibArchive REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

# Instead of finding HDF5 installed on the system, we use FetchContent
# to download and build HDF5 with thread safety enabled.
include(FetchContent)
FetchContent_Declare(
  hdf5
  URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.4/src/hdf5-1.10.4.tar.gz
)
# Set options for a thread-safe build.
set(HDF5_ENABLE_THREADSAFE ON CACHE BOOL "Enable HDF5 thread safety")
set(HDF5_USE_FILE_LOCKING FALSE CACHE BOOL "Disable HDF5 file locking")
set(HDF5_BUILD_HL OFF CACHE BOOL "Disable HDF5 HL API")
FetchContent_MakeAvailable(hdf5)
# After this call, the thread-safe HDF5 should be built and installed locally.
# Many HDF5 CMake builds provide a target called HDF5::HDF5.

# Create pybind11 modules.
pybind11_add_module(sheet_generation solver.cpp)
pybind11_add_module(pointcloud_processing pointcloud_processing.cpp)
pybind11_add_module(meshing_utils meshing_utils.cpp)

# Link libraries for each module.
target_link_libraries(sheet_generation PRIVATE yaml-cpp Eigen3::Eigen ${CMAKE_THREAD_LIBS_INIT} ${PYTHON_LIBRARIES})
target_link_libraries(pointcloud_processing PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${LibArchive_LIBRARIES} TBB::tbb hdbscan OpenMP::OpenMP_CXX ${PYTHON_LIBRARIES} HDF5::HDF5)
target_link_libraries(meshing_utils PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${LibArchive_LIBRARIES} TBB::tbb hdbscan OpenMP::OpenMP_CXX ${PYTHON_LIBRARIES})

# Define compile flags for thread-safe HDF5 usage.
target_compile_definitions(pointcloud_processing PRIVATE H5_HAVE_THREADSAFE HDF5_USE_FILE_LOCKING=FALSE)
