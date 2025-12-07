cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od")
set(VK ${CMAKE_SOURCE_DIR}/vk_backend)
set(KTX_DIR ${CMAKE_SOURCE_DIR}/ktx)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/imgui)
set(IMNODE_DIR ${CMAKE_SOURCE_DIR}/imnodes)


if (SUB_DIR)
    set(VK ${CMAKE_SOURCE_DIR}/source/core/GPU/vk_backend)
    set(KTX_DIR ${CMAKE_SOURCE_DIR}/source/core/GPU/ktx)
    set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/source/core/GPU/imgui)
    set(IMNODE_DIR ${CMAKE_SOURCE_DIR}/source/core/GPU/imnodes)
endif ()


include_directories(
        ${VK_BACKEND}/
        ${VK_BACKEND}/../../util
        ${IMGUI_DIR}/
        ${IMGUI_DIR}/back
        ${IMGUI_DIR}/back/vulkan
        ${IMNODE_DIR}/
        ${KTX_DIR}/include
        ${KTX_DIR}/lib
)

find_package(assimp REQUIRED)
find_package(Vulkan REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(unofficial-shaderc
        CONFIG REQUIRED
        PATHS "C:/Users/dlwog/vcpkg/installed/x64-windows/share/shaderc")
add_library(vk_imple STATIC
        ${VK_BACKEND}/vk_command_buffer.cpp
        ${VK_BACKEND}/vk_sync_object.cpp
        ${VK_BACKEND}/vk_context.cpp
        ${VK_BACKEND}/vk_scheduler.cpp
        ${VK_BACKEND}/vk_graph_builder.cpp
        ${VK_BACKEND}/vk_graph_compiler.cpp
        ${VK_BACKEND}/vk_resource_allocator.cpp
        ${VK_BACKEND}/vk_memory_allocator.cpp
        ${VK_BACKEND}/vk_memory_pool.cpp
        ${VK_BACKEND}/vk_graph_executer.cpp
        ${VK_BACKEND}/vk_swapchain.cpp
        ${VK_BACKEND}/vk_descriptor_allocator.cpp

        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/back/imgui_impl_vulkan.cpp
)

add_library(imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/back/imgui_impl_vulkan.cpp
)

add_library(imnode STATIC
        ${IMNODE_DIR}/imnodes.cpp)

add_library(ktx STATIC
        ${KTX_DIR}/lib/checkheader.c
        ${KTX_DIR}/lib/errstr.c
        ${KTX_DIR}/lib/etcunpack.cxx
        ${KTX_DIR}/lib/filestream.c
        ${KTX_DIR}/lib/hashlist.c
        ${KTX_DIR}/lib/hashtable.c
        ${KTX_DIR}/lib/memstream.c
        ${KTX_DIR}/lib/swap.c
        ${KTX_DIR}/lib/texture.c
        ${KTX_DIR}/lib/vk_funcs.c
        ${KTX_DIR}/lib/writer.c
        ${KTX_DIR}/lib/writer_v1.c
)

set(extern_library
        glfw
        vulkan
        assimp::assimp
        unofficial::shaderc::shaderc
)
target_include_directories(vk_imple PRIVATE  ${Vulkan_INCLUDE_DIRS})
set_target_properties(ktx PROPERTIES LINKER_LANGUAGE C)
add_definitions(-DKTX_USE_KHRONOS_API_VULKAN=1)
add_definitions(-DSPDLOG_HEADER_ONLY)
target_include_directories(vk_imple
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/VK
)


# target_compile_definitions( render_test
#         ASSET_MODELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets/models/\"
#         ASSET_TEXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/\"
#         PIPELINE_CACHE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/extern_lib/pipelineCache/\"
# )

