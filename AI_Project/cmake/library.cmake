
find_package(Vulkan REQUIRED)
set(compute ${CMAKE_CURRENT_SOURCE_DIR}/source/compute)
set(EXTERN ${CMAKE_CURRENT_SOURCE_DIR}/extern)

include_directories(
        ${Vulkan_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/source
        ${CMAKE_CURRENT_SOURCE_DIR}/source/compute
        ${CMAKE_CURRENT_SOURCE_DIR}/source/compute/vk
        ${CMAKE_CURRENT_SOURCE_DIR}/source/compute/opt
        ${CMAKE_CURRENT_SOURCE_DIR}/source/compute/vulkan-kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/source/compute/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/extern
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11
        ${CMAKE_CURRENT_SOURCE_DIR}/source/AI
        ${CMAKE_CURRENT_SOURCE_DIR}/source/python
        "C:/VulkanSDK/1.4.328.1/Include/vulkan/vulkan.hpp"
)
file(GLOB comp "source/compute/*.cpp")
file(GLOB vk "source/compute/vk/*.cpp")
file(GLOB opt "source/compute/opt/*.cpp")

add_library(mml
        ${vk}
        ${comp}
        ${opt}
        ../source/compute/model/model.cpp
)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARIES})

message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    execute_process(
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${TEST_SHADER_FILE}"
            OUTPUT_VARIABLE glslc_output
            ERROR_VARIABLE glslc_error
    )

    if (${glslc_error} MATCHES ".*extension not supported: ${EXTENSION_NAME}.*")
        message(STATUS "${EXTENSION_NAME} not supported by glslc")
        set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
    else ()
        message(STATUS "${EXTENSION_NAME} supported by glslc")
        set(${RESULT_VARIABLE} ON PARENT_SCOPE)
        add_compile_definitions(${RESULT_VARIABLE})

        # Ensure the extension support is forwarded to vulkan-kernels-gen
        list(APPEND VULKAN_KERNELS_GEN_CMAKE_ARGS -D${RESULT_VARIABLE}=ON)
        set(VULKAN_KERNELS_GEN_CMAKE_ARGS "${VULKAN_KERNELS_GEN_CMAKE_ARGS}" PARENT_SCOPE)
    endif ()
endfunction()


set(VULKAN_KERNELS_GEN_CMAKE_ARGS "")
# Test all shader extensions
test_shader_extension_support(
        "GL_KHR_cooperative_matrix"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-kernels/feature-tests/coopmat.comp"
        "V_VULKAN_COOPMAT_GLSLC_SUPPORT"
)

test_shader_extension_support(
        "GL_NV_cooperative_matrix2"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-kernels/feature-tests/coopmat2.comp"
        "V_VULKAN_COOPMAT2_GLSLC_SUPPORT"
)
test_shader_extension_support(
        "GL_EXT_integer_dot_product"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-kernels/feature-tests/integer_dot.comp"
        "V_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"
)
test_shader_extension_support(
        "GL_EXT_bfloat16"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-kernels/feature-tests/bfloat16.comp"
        "V_VULKAN_BFLOAT16_GLSLC_SUPPORT"
)
# Workaround to the "can't dereference invalidated vector iterator" bug in clang-cl debug build
# Posssibly relevant: https://stackoverflow.com/questions/74748276/visual-studio-no-displays-the-correct-length-of-stdvector
if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
endif ()

if (V_VULKAN_CHECK_RESULTS)
    add_compile_definitions(V_VULKAN_CHECK_RESULTS)
endif ()

if (V_VULKAN_DEBUG)
    add_compile_definitions(V_VULKAN_DEBUG)
endif ()

if (V_VULKAN_MEMORY_DEBUG)
    add_compile_definitions(V_VULKAN_MEMORY_DEBUG)
endif ()

if (V_VULKAN_SHADER_DEBUG_INFO)
    add_compile_definitions(V_VULKAN_SHADER_DEBUG_INFO)
    list(APPEND VULKAN_KERNELS_GEN_CMAKE_ARGS -DV_VULKAN_SHADER_DEBUG_INFO=ON)
endif ()

if (V_VULKAN_VALIDATE)
    add_compile_definitions(V_VULKAN_VALIDATE)
endif ()

if (V_VULKAN_RUN_TESTS)
    add_compile_definitions(V_VULKAN_RUN_TESTS)
endif ()

# Set up toolchain for host compilation whether cross-compiling or not
if (CMAKE_CROSSCOMPILING)
    if (V_VULKAN_KERNELS_GEN_TOOLCHAIN)
        set(HOST_CMAKE_TOOLCHAIN_FILE ${V_VULKAN_KERNELS_GEN_TOOLCHAIN})
    else ()
        detect_host_compiler()
        if (NOT HOST_C_COMPILER OR NOT HOST_CXX_COMPILER)
            message(FATAL_ERROR "Host compiler not found")
        else ()
            message(STATUS "Host compiler: ${HOST_C_COMPILER} ${HOST_CXX_COMPILER}")
        endif ()
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/host-toolchain.cmake.in ${CMAKE_BINARY_DIR}/host-toolchain.cmake @ONLY)
        set(HOST_CMAKE_TOOLCHAIN_FILE ${CMAKE_BINARY_DIR}/host-toolchain.cmake)
    endif ()
else ()
    # For non-cross-compiling, use empty toolchain (use host compiler)
    set(HOST_CMAKE_TOOLCHAIN_FILE "")
endif ()
include(ExternalProject)

if (CMAKE_CROSSCOMPILING)
    list(APPEND VULKAN_KERNELS_GEN_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${HOST_CMAKE_TOOLCHAIN_FILE})
    message(STATUS "vulkan-kernels-gen toolchain file: ${HOST_CMAKE_TOOLCHAIN_FILE}")
endif ()

ExternalProject_Add(
        vulkan-kernels-gen
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/compute/vulkan-kernels
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/$<CONFIG>
        -DCMAKE_INSTALL_BINDIR=.
        -DCMAKE_BUILD_TYPE=$<CONFIG>
        ${VULKAN_KERNELS_GEN_CMAKE_ARGS}

        BUILD_COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIG>
        BUILD_ALWAYS TRUE
        # NOTE: When DESTDIR is set using Makefile generators and
        # "make install" triggers the build step, vulkan-kernels-gen
        # would be installed into the DESTDIR prefix, so it is unset
        # to ensure that does not happen.
        INSTALL_COMMAND ${CMAKE_COMMAND} -E env --unset=DESTDIR
        ${CMAKE_COMMAND} --install . --config $<CONFIG>
)

set(_v_vk_host_suffix $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Windows>,.exe,>)
set(_v_vk_genshaders_dir "${CMAKE_BINARY_DIR}/$<CONFIG>")
set(_v_vk_genshaders_cmd "${_v_vk_genshaders_dir}/vulkan-kernels-gen${_v_vk_host_suffix}")
set(_v_vk_header "${CMAKE_CURRENT_BINARY_DIR}/v-vulkan-kernels.hpp")
set(_v_vk_input_dir "source/compute/vulkan-kernels")
set(_v_vk_output_dir "${CMAKE_CURRENT_BINARY_DIR}/vulkan-kernels-spv")

file(GLOB _v_vk_shader_files CONFIGURE_DEPENDS "${_v_vk_input_dir}/*.comp")

# Because external projects do not provide source-level tracking,
# the vulkan-kernels-gen sources need to be explicitly added to
# ensure that changes will cascade into shader re-generation.

file(GLOB _v_vk_shaders_gen_sources
        CONFIGURE_DEPENDS "${_v_vk_input_dir}/*.cpp"
        "${_v_vk_input_dir}/*.h")

add_custom_command(
        OUTPUT ${_v_vk_header}
        COMMAND ${_v_vk_genshaders_cmd}
        --output-dir ${_v_vk_output_dir}
        --target-hpp ${_v_vk_header}
        DEPENDS ${_v_vk_shaders_gen_sources}
        vulkan-kernels-gen
        COMMENT "Generate vulkan kernels header"
)

target_sources(mml PRIVATE ${_v_vk_header})
add_dependencies(mml vulkan-kernels-gen)
foreach (file_full ${_v_vk_shader_files})
    get_filename_component(file ${file_full} NAME)
    set(_v_vk_target_cpp "${CMAKE_CURRENT_BINARY_DIR}/${file}.cpp")
    add_custom_command(
            OUTPUT ${_v_vk_target_cpp}
            DEPFILE ${_v_vk_target_cpp}.d
            COMMAND ${_v_vk_genshaders_cmd}
            --glslc ${Vulkan_GLSLC_EXECUTABLE}
            --source ${file_full}
            --output-dir ${_v_vk_output_dir}
            --target-hpp ${_v_vk_header}
            --target-cpp ${_v_vk_target_cpp}
            DEPENDS ${file_full}
            ${_v_vk_shaders_gen_sources}
            vulkan-kernels-gen
            COMMENT "Generate vulkan kernels for ${file}"
    )
    target_sources(mml PRIVATE ${_v_vk_target_cpp})
endforeach ()
target_include_directories(mml PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${_v_vk_output_dir}
)






