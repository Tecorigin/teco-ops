cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_C_COMPILER "nvcc")
set(CMAKE_CXX_COMPILER "nvcc")
set(CMAKE_CUDA_ARCHITECTURES "89")
project(cuda_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcompiler -fPIC")

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
add_definitions(-DUSE_CUDA)

add_library(ops_objs OBJECT ${SRC})
set_target_properties(ops_objs PROPERTIES
    CUDA_ARCHITECTURES "89"
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(ops_objs PRIVATE -O3 --extended-lambda)
target_include_directories(ops_objs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")
target_include_directories(ops_objs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(cuda_ops SHARED $<TARGET_OBJECTS:ops_objs>)
