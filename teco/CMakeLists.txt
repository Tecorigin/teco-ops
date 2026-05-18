cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_C_COMPILER "tecocc")
set(CMAKE_CXX_COMPILER "tecocc")

project(teco_ops)

set(PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(-Werror)

option(WITH_TORCH "Enable PyTorch support" OFF)
if(WITH_TORCH)
    find_package(Python3 REQUIRED COMPONENTS Development)
    find_package(Torch REQUIRED)
    add_definitions(-DWITH_TORCH)
    include_directories(${TORCH_INCLUDE_DIRS})
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${TORCH_SDAA_HOME}/include)
    get_filename_component(TORCH_LIB_DIR ${TORCH_LIBRARY} DIRECTORY)
    message(STATUS "WITH_TORCH is enabled")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "TORCH_LIB_DIR: ${TORCH_LIB_DIR}")
else()
    message(STATUS "WITH_TORCH is disabled")
endif()

set(TECOOPS_OPS_PATH ${OPS_DIR}/interface/ops)

file(GLOB_RECURSE SRC_INTERFACE_COMMON "${OPS_DIR}/interface/common/*.cpp")
file(GLOB_RECURSE SRC_INTERFACE_OPS "${OPS_DIR}/interface/ops/*.cpp")
file(GLOB_RECURSE SRC_UAL_COMMON "${OPS_DIR}/ual/com/*.cpp")
file(GLOB_RECURSE SRC_UAL_OPS "${OPS_DIR}/ual/ops/*/*.cpp")
file(GLOB_RECURSE SRC_UAL_KERNEL "${OPS_DIR}/ual/kernel/*/*.scpp")

list(APPEND SRC_ALL
    ${SRC_INTERFACE_COMMON}
    ${SRC_INTERFACE_OPS}
    ${SRC_UAL_COMMON}
    ${SRC_UAL_OPS}
    ${SRC_UAL_KERNEL})

set(complie_options -O3 -msimd -mslave -fPIC -flto -x sdaa -std=c++17)
add_definitions(-DUSE_TECO)

add_library(ops_objs OBJECT ${SRC_ALL})
set_source_files_properties(${SRC_UAL_KERNEL} PROPERTIES LANGUAGE CXX)
set_target_properties(ops_objs PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(ops_objs PRIVATE ${complie_options})
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/../common")
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/")

if(WITH_TORCH)
    set(TORCH_LINK_FLAGS "-L${TORCH_LIB_DIR} -ltorch -ltorch_cpu -lc10")
else()
    set(TORCH_LINK_FLAGS "")
endif()

add_custom_target(ops_lib ALL
        COMMAND mkdir -p ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        COMMAND tecocc $<TARGET_OBJECTS:ops_objs> -flto -ffp-contract=fast -fPIC -shared --sdaa-link -fuse-ld=lld
        ${TORCH_LINK_FLAGS}
        -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libteco_ops.so
        COMMAND_EXPAND_LISTS
)

add_dependencies(ops_lib ops_objs)
set(CMAKE_CXX_LINK_FLAGS "-L$ENV{SDAA_HOME}/lib64  -lsdaart")