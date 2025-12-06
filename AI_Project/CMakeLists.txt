
cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-DSPDLOG_COMPILED_LIB)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
    add_compile_options(/utf-8)
    add_compile_options(
            /wd4018
            /wd4267
            /wd2666
    )
    execute_process(COMMAND chcp 65001)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif ()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/source
        ${CMAKE_CURRENT_SOURCE_DIR}/source/compute
        ${CMAKE_CURRENT_SOURCE_DIR}/source/compute/vulkan-kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/source/compute/ggml-cpu
        ${Vulkan_INCLUDE_DIRS}
        "C:/VulkanSDK/1.4.328.1/Include/vulkan/vulkan.hpp"
        ${CMAKE_CURRENT_SOURCE_DIR}/extern
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11
        ${CMAKE_CURRENT_SOURCE_DIR}/source/AI
        ${CMAKE_CURRENT_SOURCE_DIR}/source/python
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/library.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/execute.cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")




