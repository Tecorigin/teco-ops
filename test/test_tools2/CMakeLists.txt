cmake_minimum_required(VERSION 3.12)

if(BUILD_SPECIFIC_OP)
    foreach(ITEM ${BUILD_SPECIFIC_OP})
        message(STATUS "BUILD_SPECIFIC_OP: ${ITEM}")
    endforeach()
endif()

set (CMAKE_C_COMPILER "/usr/bin/gcc")
set (CMAKE_CXX_COMPILER "/usr/bin/g++")

project(prototxt2pb)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(THIRDPARTY_DIR ${PROJECT_SOURCE_DIR}/../thirdparty)
include_directories(${THIRDPARTY_DIR}/protobuf/include)
link_directories(${THIRDPARTY_DIR}/protobuf/lib)
link_directories(${THIRDPARTY_DIR}/protobuf/lib64)

set(TEST_PROTO_INCLUDE ${PROJECT_SOURCE_DIR}/../test_proto)
file(GLOB_RECURSE PROTO_SOURCE ${PROJECT_SOURCE_DIR}/../test_proto/*.cc)
include_directories(${TEST_PROTO_INCLUDE})

add_executable(${PROJECT_NAME} prototxt2pb.cpp ${PROTO_SOURCE}) 
target_link_libraries(${PROJECT_NAME} libprotobuf.a stdc++ dl stdc++fs rt pthread)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
