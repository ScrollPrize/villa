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
find_package(Threads REQUIRED)  # Find the threading library
find_package(LibArchive REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
# Find the HighFive library if it's installed on the system
find_package(HDF5 REQUIRED COMPONENTS C CXX HL) # HDF5 is a dependency for HighFive

pybind11_add_module(sheet_generation solver.cpp)
pybind11_add_module(pointcloud_processing pointcloud_processing.cpp)
pybind11_add_module(meshing_utils meshing_utils.cpp)
# Link YAML-CPP library
target_link_libraries(sheet_generation PRIVATE yaml-cpp Eigen3::Eigen ${CMAKE_THREAD_LIBS_INIT} ${PYTHON_LIBRARIES})
target_link_libraries(pointcloud_processing PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${LibArchive_LIBRARIES} TBB::tbb hdbscan OpenMP::OpenMP_CXX ${PYTHON_LIBRARIES} HDF5::HDF5)
target_link_libraries(meshing_utils PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${LibArchive_LIBRARIES} TBB::tbb hdbscan OpenMP::OpenMP_CXX ${PYTHON_LIBRARIES})