# Install script for directory: C:/CSC305/Assignemnt4/bundle/build/_deps/atlas-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/csc305_assignments")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/atlas-build/external/imgui/cmake_install.cmake")
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/atlas-build/external/stb/cmake_install.cmake")
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/atlas-build/include/atlas/cmake_install.cmake")
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/atlas-build/source/atlas/cmake_install.cmake")
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/fmt-build/cmake_install.cmake")
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/magic_enum-build/cmake_install.cmake")
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/glm-build/cmake_install.cmake")
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/glfw-build/cmake_install.cmake")
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/gl3w-build/cmake_install.cmake")
  include("C:/CSC305/Assignemnt4/bundle/build/_deps/tinyobjloader-build/cmake_install.cmake")

endif()

