cmake_minimum_required(VERSION 3.10)
project(nn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
include_directories(/usr/include)

option(BUILD_DEBUG "Enable debug mode" OFF)
if(BUILD_DEBUG)
    message(STATUS "Building with debug mode enabled")
    add_definitions(-DDEBUG)
endif()

add_executable(nn main.cu)

target_link_libraries(nn ${OpenCV_LIBS} nvinfer nvonnxparser cuda cudart pthread rt)
