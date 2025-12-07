cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(SPDLOG_COMPILED_LIB ON)
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

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/source)
set(EXTERN_DIR ${CMAKE_SOURCE_DIR}/extern)
set(GLM_DIR ${CMAKE_SOURCE_DIR}/extern/glm)
set(EXTERN_DIR ${CMAKE_SOURCE_DIR}/extern)
set(MODEL_DIR ${EXTERN_DIR}/externModel/assets/models/)
set(TEXTURE_DIR ${EXTERN_DIR}/externModel/assets/textures)
set(SCENE ${CMAKE_SOURCE_DIR}/source/core/scene_graph)
set(IM_NODES ${CMAKE_SOURCE_DIR}/source/core/GPU/imnodes/)
set(RENDER ${CMAKE_SOURCE_DIR}/source/core/render)

add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
add_compile_options(/wd4005 /wd4996 /wd4834)
include_directories(
        ${SOURCE_DIR}/resource
        ${SOURCE_DIR}/core/resource
        ${SOURCE_DIR}/
        ${SOURCE_DIR}/renderer
        ${SOURCE_DIR}/model
        ${SOURCE_DIR}/core
        ${SOURCE_DIR}/core/GPU
        ${SOURCE_DIR}/core/GPU/VK
        ${SOURCE_DIR}/core/util
        ${SOURCE_DIR}/core/io
        ${SOURCE_DIR}/scene_graph

        #${SOURCE_DIR}/feature/sculptor
        ${SOURCE_DIR}/feature/UI

        ${EXTERN_DIR}
        ${EXTERN_DIR}/CubismSdkForNative-5-r.4.1
        ${EXTERN_DIR}/CubismSdkForNative-5-r.4.1/Core
        ${EXTERN_DIR}/CubismSdkForNative-5-r.4.1/Framework
        ${EXTERN_DIR}/CubismSdkForNative-5-r.4.1/Samples

        ${EXTERN_DIR}/imgui
        ${EXTERN_DIR}/imgui/back

        ${EXTERN_DIR}/ktx/lib
        ${EXTERN_DIR}/ktx/include
        ${EXTERN_DIR}/ktx/other_include

)

find_package(assimp REQUIRED)
find_package(spdlog REQUIRED)
find_package(Vulkan REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(unofficial-shaderc CONFIG REQUIRED PATHS "C:/Users/dlwog/vcpkg/installed/x64-windows/share/shaderc")

option(SUB_DIR "SUB_DIR" ON)

set(VK_BACKEND ${CMAKE_SOURCE_DIR}/source/core/GPU/VK)
set(KTX_DIR ${CMAKE_SOURCE_DIR}/source/core/GPU/ktx)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/source/core/GPU/imgui)
set(IMNODE_DIR ${CMAKE_SOURCE_DIR}/source/core/GPU/imnodes)

add_definitions(-DSPDLOG_HEADER_ONLY)
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
# 공통 소스 파일
set(COMMON_SRC
        main.cpp
        source/core/engine.cpp
        source/core/GPU/gpu_context.cpp
        source/core/resource/resource_manager.cpp
        source/core/resource/importer.cpp

        source/core/io/io.cpp
        source/core/io/log_sink.cpp
        source/core/io/event_manager.cpp
        #source/core/scene_graph/mesh.cpp
        #source/feature/sculptor/sculptor.cpp
        #source/feature/sculptor/dynamic_mesh.cpp
        #source/feature/sculptor/sculptor_act.cpp

        source/core/UI/ui.cpp

        ${VK_BACKEND}/vk_pass.cpp
        ${VK_BACKEND}/vk_shader_pool.cpp
        ${VK_BACKEND}/vk_command_buffer.cpp
        ${VK_BACKEND}/vk_sync_object.cpp
        ${VK_BACKEND}/vk_context.cpp
        ${VK_BACKEND}/vk_scheduler.cpp
        ${VK_BACKEND}/vk_graph_builder.cpp
        ${VK_BACKEND}/vk_resource_allocator.cpp
        ${VK_BACKEND}/vk_memory_allocator.cpp
        ${VK_BACKEND}/vk_memory_pool.cpp
        ${VK_BACKEND}/vk_resource.cpp
        ${VK_BACKEND}/vk_swapchain.cpp
        ${VK_BACKEND}/vk_pipeline_pool.cpp
        ${VK_BACKEND}/vk_descriptor_allocator.cpp
        ${VK_BACKEND}/vk_sampler_builder.cpp
        ${VK_BACKEND}/vk_texture.cpp
        ${VK_BACKEND}/vk_discard_pool.cpp
        ${VK_BACKEND}/vk_resource.cpp
        ${VK_BACKEND}/vk_descriptor_layout_builder.cpp
        ${VK_BACKEND}/vk_host_buffer.cpp


        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/back/imgui_impl_vulkan.cpp
        ${IM_NODES}/imnodes.cpp

        ${SCENE}/model.cpp
        ${SCENE}/camera_func.cpp
        ${SCENE}/camera_state.cpp
        ${SCENE}/light.cpp

        ${RENDER}/frame_graph.cpp
        ${RENDER}/render_pipeline.cpp
        ${RENDER}/RenderPassPool.cpp
        ${RENDER}/render_target_film.cpp
        ${RENDER}/frame_graph_viewer.cpp
        ${RENDER}/renderer.cpp

)
message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
add_executable(exe ${COMMON_SRC})
# OS별 링크 및 정의
if (WIN32)
    target_link_libraries(exe
            glfw
            Vulkan::Vulkan
            spdlog::spdlog
            spdlog::spdlog_header_only
            fmt::fmt
            assimp::assimp
            unofficial::shaderc::shaderc
    )
else ()
    target_link_libraries(exe
            glfw
            vulkan
            vk_imple
            dl
            pthread
            X11
            Xxf86vm
            Xrandr
            Xi
            shaderc
            spdlog::spdlog
            assimp::assimp
            imnode
            ktx
    )
endif ()


target_include_directories(exe PRIVATE ${Vulkan_INCLUDE_DIRS})
target_compile_definitions(exe PRIVATE
        ASSET_MODELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets/models/\"
        ASSET_TEXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/\"
        PIPELINE_CACHE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/extern/pipelineCache/\"
        SPDLOG_HEADER_ONLY
        BUILD_SHARED_LIBS=OFF/SPDLOG_BUILD_SHARED=OFF
)

target_compile_options(exe PRIVATE /W0)
target_compile_definitions(exe PRIVATE FORCE_UTF8_CONSOLE)
