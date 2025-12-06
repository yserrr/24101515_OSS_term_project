# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp"
  "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-build"
  "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix"
  "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/tmp"
  "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp"
  "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src"
  "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ljh/workspace/opengl_course-main/projects/13_advanced_lighting/build/dep_assimp-prefix/src/dep_assimp-stamp${cfgdir}") # cfgdir has leading slash
endif()
