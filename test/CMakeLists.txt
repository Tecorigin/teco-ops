cmake_minimum_required(VERSION 3.12)

project(demo)

set(CMAKE_CXX_STANDARD 14) # GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THIRDPARTY_DIR ${PROJECT_SOURCE_DIR}/thirdparty)
include_directories(${THIRDPARTY_DIR}/protobuf/include)
link_directories(${THIRDPARTY_DIR}/protobuf/lib)
link_directories(${THIRDPARTY_DIR}/protobuf/lib64)
include_directories(${THIRDPARTY_DIR}/googletest/include)
link_directories(${THIRDPARTY_DIR}/googletest/lib)
link_directories(${THIRDPARTY_DIR}/googletest/lib64)

enable_testing()
include(GoogleTest)

option(USE_TECO "teco device option" ON)
option(USE_CUDA "cuda device option" OFF)

if(USE_TECO)
    add_definitions(-DUSE_TECO)
    set(TARGET_NAME demo)
elseif(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    set(TARGET_NAME demo_cuda)
endif()

if(BUILD_SPECIFIC_OP)
    foreach(ITEM ${BUILD_SPECIFIC_OP})
        message(STATUS "BUILD_SPECIFIC_OP: ${ITEM}")
    endforeach()
endif()

#execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_proto/type_convert.py)
find_package(Python3 COMPONENTS Interpreter)
find_package(PythonInterp)

if(Python3_FOUND)
    if(BUILD_SPECIFIC_OP)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/register_op.py zoo teco ${BUILD_SPECIFIC_OP}
        )
    else()
        execute_process(
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/register_op.py zoo teco
        )
    endif()
elseif(PYTHONINTERP_FOUND)
    if(BUILD_SPECIFIC_OP)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/register_op.py zoo teco ${BUILD_SPECIFIC_OP}
        )
    else()
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/register_op.py zoo teco
        )
    endif()
else()
    message(FATAL_ERROR "Neither Python3 nor Python2 interpreter found. Please install Python to continue.")
endif()

#set source
file(GLOB_RECURSE PROTO_SOURCE ${PROJECT_SOURCE_DIR}/test_proto/*.cc)
file(GLOB_RECURSE SRC_SOURCE ${PROJECT_SOURCE_DIR}/src/*.cpp)
if(USE_TECO)
    file(GLOB_RECURSE DEVICE_SOURCE ${PROJECT_SOURCE_DIR}/src/device/cuda/*.cpp)
    list(REMOVE_ITEM SRC_SOURCE ${DEVICE_SOURCE})
elseif(USE_CUDA)
    file(GLOB_RECURSE DEVICE_SOURCE ${PROJECT_SOURCE_DIR}/src/device/sdaa/*.cpp)
    list(REMOVE_ITEM SRC_SOURCE ${DEVICE_SOURCE})
endif()

set(TECOAL_TESTSUITE_SOURCE)
set(CASE_PATH ${PROJECT_SOURCE_DIR}/zoo/teco)
file(GLOB TECOAL_CASE_SOURCE ${CASE_PATH}/*.cpp)
if(BUILD_SPECIFIC_OP)
    foreach(SPECIFIC_OP ${BUILD_SPECIFIC_OP})
        file(GLOB_RECURSE OP_SOURCE "${CASE_PATH}/${SPECIFIC_OP}/*.cpp")
        list(APPEND TECOAL_TESTSUITE_SOURCE ${OP_SOURCE})
    endforeach()
else()
    file(GLOB_RECURSE TECOAL_TESTSUITE_SOURCE "${CASE_PATH}/*.cpp")
endif()

set(PROJECT_INCLUDE ${PROJECT_SOURCE_DIR})
include_directories(
    ${PROJECT_INCLUDE}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/test_proto)
include_directories(${PROJECT_SOURCE_DIR}/../common)

add_executable(${TARGET_NAME} 
               main.cpp 
               ${SRC_SOURCE}
               ${PROTO_SOURCE}
               ${TECOAL_CASE_SOURCE}
               ${TECOAL_TESTSUITE_SOURCE})

target_link_libraries(${TARGET_NAME} stdc++fs)
target_link_libraries(${TARGET_NAME} libprotobuf.a libgtest.a)
target_link_libraries(${TARGET_NAME} stdc++ dl stdc++fs rt pthread)

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 14)


if(USE_TECO)

    set(COM_INCLUDE /opt/tecoai/include)
    set(COM_LIB /opt/tecoai/lib64)
    if(TECO_HOME)
        set(TECO_INCLUDE ${TECO_HOME}/include)
        set(TECO_LIB ${TECO_HOME}/lib)
        set(TECO_LIB64 ${TECO_HOME}/lib64)
    else()
        set(TECO_INCLUDE ${PROJECT_SOURCE_DIR}/../teco/interface/include)
        set(TECO_LIB ${PROJECT_SOURCE_DIR}/../build/teco/lib)
        set(TECO_LIB64 ${PROJECT_SOURCE_DIR}/../build/teco/lib64)
    endif()

    include_directories(AFTER
        ${TECO_INCLUDE}
    )

    include_directories(
        ${COM_INCLUDE}
    )

    link_directories (AFTER
        ${TECO_LIB}
        ${TECO_LIB64}
    )

    link_directories(
        ${COM_LIB}
    )

    include_directories(${PROJECT_SOURCE_DIR}/../teco)

find_library(SDAART_LIBRARY NAMES sdaart PATHS /opt/tecoai/lib64)
target_link_libraries(${TARGET_NAME} ${SDAART_LIBRARY})
find_library(SDPTI_LIBRARY NAMES sdpti PATHS /opt/tecoai/lib64)
target_link_libraries(${TARGET_NAME} ${SDPTI_LIBRARY})
set(TECOAL_LIB ${PROJECT_SOURCE_DIR}/../build/teco/lib)
find_library(TECOAL_LIBRARY NAMES teco_ops PATHS ${TECOAL_LIB})
target_link_libraries(${TARGET_NAME} ${TECOAL_LIBRARY})

elseif(USE_CUDA)
    set(CUDA_PATH /usr/local/cuda)
    set(CUDA_INCLUDE ${CUDA_PATH}/include)
    set(CUDA_LIB ${CUDA_PATH}/lib64)
    set(CUPTI_INCLUDE ${CUDA_PATH}/extras/CUPTI/include)
    set(CUPTI_LIB ${CUDA_PATH}/extras/CUPTI/lib64)

    include_directories(
        ${CUDA_INCLUDE}
        ${CUPTI_INCLUDE}
    )

    link_directories(
        ${CUDA_LIB}
        ${CUPTI_LIB}
    )

    # target_link_libraries(${TARGET_NAME} cupti cudart cuda)

find_library(SDAART_LIBRARY NAMES cudart PATHS /usr/local/cuda/lib64)
target_link_libraries(${TARGET_NAME} ${SDAART_LIBRARY})
find_library(SDAA_LIBRARY NAMES cuda PATHS /usr/local/cuda/lib64)
target_link_libraries(${TARGET_NAME} ${SDAA_LIBRARY})
find_library(SDPTI_LIBRARY NAMES cupti PATHS /usr/local/cuda/extras/CUPTI/lib64)
target_link_libraries(${TARGET_NAME} ${SDPTI_LIBRARY})
    include_directories(${PROJECT_SOURCE_DIR}/../cuda)
    set(TECOAL_LIB ${PROJECT_SOURCE_DIR}/../build/cuda/lib)
    find_library(TECOAL_LIBRARY NAMES cuda_ops PATHS ${TECOAL_LIB})
    target_link_libraries(${TARGET_NAME} ${TECOAL_LIBRARY})
endif()
