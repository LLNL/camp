cmake_minimum_required (VERSION 3.9)

project (camp
  LANGUAGES CXX
  VERSION 0.1.0)

# Default to C++11 if not set so GTest/GMock can build
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++14" CACHE STRING "")
else()
    if (${BLT_CXX_STD} STREQUAL "c++11")
      message(FATAL_ERROR "CAMP and the RAJA framework no longer support c++11, select a c++ standard of 14 or higher")
    endif()
endif()

include(cmake/load_blt.cmake)


cmake_dependent_option(CAMP_ENABLE_TESTS "Build tests" On "ENABLE_TESTS" Off)

# if ENABLE_TESTS is defined by a parent project, and
# CAMP_ENABLE_TESTS has not been set to OFF set the
# value of CAMP_ENABLE_TESTS to the value of ENABLE_TESTS.
if (CAMP_ENABLE_TESTS AND DEFINED ENABLE_TESTS)
  set(CAMP_ENABLE_TESTS ${ENABLE_TESTS})
endif()

if (WIN32)
  # use git-bash for windows, wsl is not populated on azure
  set(BASH "C:/Program Files/Git/bin/bash.exe")
else()
  set(BASH "bash")
endif()
# generate list of headers at configure time, no this is not perfect
# note that this *DOES* work on windows if bash is installed, which
# it is on azure
execute_process(COMMAND ${BASH} ${PROJECT_SOURCE_DIR}/scripts/gen-header-list.sh
  OUTPUT_FILE ${PROJECT_BINARY_DIR}/camp_headers.cmake
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
execute_process(COMMAND ${BASH} ${PROJECT_SOURCE_DIR}/scripts/gen-header-list.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
include(${PROJECT_BINARY_DIR}/camp_headers.cmake)

# backends
set (camp_depends)

set (camp_backends openmp cuda hip sycl)

foreach (backend ${camp_backends})
  string(TOUPPER "${backend}" suffix)
  if ("${ENABLE_${suffix}}")
    set ("CAMP_ENABLE_${suffix}" On)
    list (APPEND camp_depends ${backend})
  endif()
endforeach()

if (ENABLE_CUDA)
  if("${CUDA_VERSION_STRING}" VERSION_LESS "9.2")
    message(FATAL_ERROR "Trying to use CUDA version ${CUDA_VERSION_STRING}. CAMP dependency Googletest requires CUDA version 9.2.x or newer.")
  endif()
  if(ENABLE_NV_TOOLS_EXT)
    set(camp_depends
      ${camp_depends}
      nvtoolsext)
  endif ()
endif ()

if (ENABLE_HIP)
  if("${HIP_VERSION_STRING}" VERSION_LESS "3.5")
    message(FATAL_ERROR "Trying to use HIP/ROCm version ${HIP_VERSION_STRING}. CAMP requires HIP/ROCm version 3.5 or newer. ")
  endif()
endif ()

# end backends

# Configure the config header file to allow config time options
configure_file(${PROJECT_SOURCE_DIR}/include/camp/defines.in.hpp
               ${PROJECT_BINARY_DIR}/include/camp/defines.hpp)

blt_add_library (
  NAME camp
  HEADERS ${camp_headers}
  DEPENDS_ON ${camp_depends}
  )
target_include_directories (camp INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )
set_target_properties (camp PROPERTIES
  INTERFACE_LIB_VERSION $camp_VERSION
  INTERFACE_COMPILE_FEATURES cxx_std_14)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/campConfigVersion.cmake"
  VERSION 0.2.3
  COMPATIBILITY AnyNewerVersion
  )

install(TARGETS camp
  EXPORT campTargets
  LIBRARY DESTINATION lib COMPONENT Runtime
  ARCHIVE DESTINATION lib COMPONENT Development
  RUNTIME DESTINATION bin COMPONENT Runtime
  PUBLIC_HEADER DESTINATION include COMPONENT Development
  BUNDLE DESTINATION bin COMPONENT Runtime
  )

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/campConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/campConfig.cmake"
  INSTALL_DESTINATION
  lib/cmake/camp
  )

install(EXPORT campTargets
  DESTINATION lib/cmake/camp)
install(FILES
  "${PROJECT_BINARY_DIR}/campConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/campConfig.cmake"
  DESTINATION
  lib/cmake/camp)
install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/include/
  DESTINATION
  include)

if(CAMP_ENABLE_TESTS)
enable_testing()
add_subdirectory(test)
endif()

