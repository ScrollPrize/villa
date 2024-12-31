cmake_minimum_required(VERSION 3.27)
project(VWTChunkloadingTestingAndBenching)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -Wall -pedantic")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)
find_package(OpenCV REQUIRED)
find_package(llfio CONFIG REQUIRED)
find_package(OpenSSL 3.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${llfio_INCLUDE_DIRS})
include_directories(${OpenSSL_INCLUDE_DIRS})

set(LLFIO_HEADERS_ONLY 1)

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "include/*.h" "src/*.cpp")

add_executable(VWTChunkloadingTestingAndBenching bench.cpp ${SRC}
        src/test/data-generator/continuous_data_generator.cpp
        include/test/data.h
        include/test/data-generator/continuous_data_generator.h)
target_include_directories(VWTChunkloadingTestingAndBenching PRIVATE include)

target_link_libraries(VWTChunkloadingTestingAndBenching PUBLIC ${OpenCV_LIBS} ${OpenSSL_LIBS} llfio::hl benchmark::benchmark Threads::Threads)
target_link_libraries(VWTChunkloadingTestingAndBenching PUBLIC OpenSSL::Crypto OpenSSL::SSL)