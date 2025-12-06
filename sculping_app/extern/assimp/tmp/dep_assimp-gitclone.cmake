# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp/dep_assimp-gitclone-lastrun.txt" AND EXISTS "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp/dep_assimp-gitinfo.txt" AND
  "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp/dep_assimp-gitclone-lastrun.txt" IS_NEWER_THAN "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp/dep_assimp-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp/dep_assimp-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --depth 1 --no-single-branch --config "advice.detachedHead=false" "https://github.com/assimp/assimp" "dep_assimp"
    WORKING_DIRECTORY "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/assimp/assimp'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "v5.0.1" --
  WORKING_DIRECTORY "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v5.0.1'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp/dep_assimp-gitinfo.txt" "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp/dep_assimp-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp/dep_assimp-gitclone-lastrun.txt'")
endif()
