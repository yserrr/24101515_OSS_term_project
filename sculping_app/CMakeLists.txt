cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif ()


include(FetchContent)

FetchContent_Declare(
        assimp
        GIT_REPOSITORY https://github.com/assimp/assimp.git
        GIT_TAG v5.2.5  #
)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(glfw)
FetchContent_MakeAvailable(assimp)
# 디렉토리 설정
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/source)
set(EXTERN_DIR ${CMAKE_SOURCE_DIR}/extern)
set(GLM_DIR ${CMAKE_SOURCE_DIR}/extern/glm)
set(EXTERN_DIR ${CMAKE_SOURCE_DIR}/extern)
set(MODEL_DIR ${EXTERN_DIR}/externModel/assets/models/)
set(TEXTURE_DIR ${EXTERN_DIR}/externModel/assets/textures)

include_directories(
        ${SOURCE_DIR}/resource
        ${SOURCE_DIR}/renderer
        ${SOURCE_DIR}/model
        ${SOURCE_DIR}/core
        ${SOURCE_DIR}/sculptor

        ${EXTERN_DIR}
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
pkg_check_modules(SHADERC_PC REQUIRED shaderc)

add_definitions(-DKTX_USE_KHRONOS_API_VULKAN=1)
add_library(ktx STATIC
        ${EXTERN_DIR}/ktx/lib/checkheader.c
        ${EXTERN_DIR}/ktx/lib/errstr.c
        ${EXTERN_DIR}/ktx/lib/etcunpack.cxx
        ${EXTERN_DIR}/ktx/lib/filestream.c
        ${EXTERN_DIR}/ktx/lib/hashlist.c
        ${EXTERN_DIR}/ktx/lib/hashtable.c
        ${EXTERN_DIR}/ktx/lib/memstream.c
        ${EXTERN_DIR}/ktx/lib/swap.c
        ${EXTERN_DIR}/ktx/lib/texture.c
        ${EXTERN_DIR}/ktx/lib/vk_funcs.c
        ${EXTERN_DIR}/ktx/lib/writer.c
        ${EXTERN_DIR}/ktx/lib/writer_v1.c
)
set_target_properties(ktx PROPERTIES LINKER_LANGUAGE C)

# 공통 소스 파일
set(COMMON_SRC
        main.cpp

        source/core/app.cpp
        source/core/engine_init.cpp
        source/core/vk_context_load.cpp
        source/core/resource_pool_init.cpp
        source/core/resource_pool.cpp
        source/core/event_manager.cpp
        source/core/renderer.cpp
        source/core/renderer_draw.cpp
        source/core/ui_renderer_draw.cpp
        source/resource/command_pool_manager.cpp
        source/resource/descriptor_manager.cpp
        source/resource/descriptor_pool.cpp
        source/resource/shader_pool.cpp
        source/resource/frame_pool.cpp
        source/resource/memory_pool.cpp
        source/resource/memory_allocator.cpp
        source/resource/static_buffer.cpp
        source/resource/descriptor_uploader.cpp
        source/resource/buffer_builder.cpp
        source/resource/swapchain_view.cpp
        source/resource/texture.cpp
        source/model/camera.cpp
        source/model/importer.cpp
        source/model/mesh.cpp
        source/resource/swapchain.cpp
        source/resource/semaphore_pool.cpp
        source/resource/pipeline_pool.cpp

        source/core/sculptor/sculptor.cpp
        source/core/sculptor/dynamic_mesh.cpp
        source/core/sculptor/sculptor_act.cpp

        extern/imgui/imgui_impl_glfw.cpp
        extern/imgui/imgui.cpp
        extern/imgui/imgui_draw.cpp
        extern/imgui/imgui_demo.cpp
        extern/imgui/imgui_tables.cpp
        extern/imgui/imgui_widgets.cpp
        extern/imgui/back/imgui_impl_vulkan.cpp


        source/resource/renderpass_pool.cpp
        source/resource/command_pool.cpp
        source/resource/semaphore_pool.cpp
        source/resource/fence_pool.cpp
        source/resource/sampler_builder.cpp
        source/core/log_sink.cpp
        source/core/formater.cpp
)

add_executable(exe ${COMMON_SRC})

# OS별 링크 및 정의
if (WIN32)
    target_link_libraries(exe
            glfw
            vulkan-1
            spdlog::spdlog
            assimp::assimp
            shaderc
            ktx
    )
else ()
    target_link_libraries(exe
            glfw
            vulkan
            dl
            pthread
            X11
            Xxf86vm
            Xrandr
            Xi
            spdlog::spdlog
            assimp::assimp
            shaderc
            ktx
    )
endif ()


target_compile_definitions(exe PRIVATE
        ASSET_MODELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets/models/\"
        ASSET_TEXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/\"
        PIPELINE_CACHE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/extern/pipelineCache/\"
)
