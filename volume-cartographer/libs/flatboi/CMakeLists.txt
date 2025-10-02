cmake_minimum_required(VERSION 3.16)
project(flatboi LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# 1) Enable PaStiX in libigl BEFORE add_subdirectory
set(LIBIGL_WITH_PASTIX ON CACHE BOOL "" FORCE)

# If you also want CHOLMOD available in the same build
# set(LIBIGL_WITH_CHOLMOD ON CACHE BOOL "" FORCE)

# 2) Bring libigl in (adjust path to your checkout)
add_subdirectory(/src/libs/libigl ${CMAKE_BINARY_DIR}/libigl-build)

# 3) Your executable
add_executable(flatboi flatboi.cpp)

# 4) Link: core + PaStiX (this pulls includes & libs via igl::pastix)
target_link_libraries(flatboi PRIVATE igl::core igl::pastix)

# 5) OpenMP is useful for PaStiX
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(flatboi PRIVATE OpenMP::OpenMP_CXX)
endif()
find_package(OpenCV REQUIRED COMPONENTS imgcodecs core)
target_link_libraries(flatboi PRIVATE ${OpenCV_LIBS})