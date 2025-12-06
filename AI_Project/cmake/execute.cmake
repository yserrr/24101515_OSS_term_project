file(GLOB SRC_FILES "source/test/*.cpp")

foreach (SRC ${SRC_FILES})
    get_filename_component(EXE_NAME ${SRC} NAME_WE)
    add_executable(${EXE_NAME} ${SRC})
    target_link_libraries(${EXE_NAME} PRIVATE
            mml
            ${Python3_LIBRARIES}
            Vulkan::Vulkan)
endforeach ()








