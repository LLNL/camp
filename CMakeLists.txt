cmake_minimum_required (VERSION 3.9)

project (camp
  LANGUAGES CXX C
  VERSION 0.4.0)

set(camp_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(camp_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(camp_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# Default to C++11 if not set so GTest/GMock can build
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++14" CACHE STRING "")
else()
    set(_unsupported_cxx "c++98" "c++11")
    if (BLT_CXX_STD IN_LIST _unsupported_cxx)
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
include(${PROJECT_BINARY_DIR}/camp_headers.cmake)
list (APPEND camp_headers ${PROJECT_BINARY_DIR}/include/camp/config.hpp)

# backends
set (camp_depends)

set (camp_backends openmp cuda blt::hip sycl)
# NOTE: camp itself doesn't require a hip or cuda compiler,
#       but some of its features are activated by having a device compiler
set (camp_runtime_backends cuda blt::hip)

foreach (backend ${camp_backends})
  string(TOUPPER "${backend}" suffix)
  if ("${ENABLE_${suffix}}")
    set ("CAMP_ENABLE_${suffix}" On)
    if (backend IN_LIST camp_runtime_backends)
      list (APPEND camp_depends ${backend}_runtime)
    else()
      list (APPEND camp_depends ${backend})
    endif()
  endif()
endforeach()

if (CAMP_ENABLE_TARGET_OPENMP)
  if (NOT CAMP_ENABLE_OPENMP)
    message(FATAL_ERROR "OpenMP must be enabled if TARGET_OPENMP is enabled")
  endif ()
endif ()

if (ENABLE_CUDA)
  if("${CUDA_VERSION_STRING}" VERSION_LESS "10.1")
    message(FATAL_ERROR "Trying to use CUDA version ${CUDA_VERSION_STRING}. CAMP requires CUDA version 10.1 or newer.")
  endif()

  if(ENABLE_NV_TOOLS_EXT)
    set(camp_depends
      ${camp_depends}
      nvtoolsext)
  endif ()
endif ()

if (ENABLE_HIP)
  if("${hip_VERSION}" VERSION_LESS "3.5")
    message(FATAL_ERROR "Trying to use HIP/ROCm version ${HIP_VERSION_STRING}. CAMP requires HIP/ROCm version 3.5 or newer. ")
  endif()
endif ()

# end backends

# Configure the config header file to allow config time options
configure_file(${PROJECT_SOURCE_DIR}/include/camp/config.in.hpp
               ${PROJECT_BINARY_DIR}/include/camp/config.hpp)

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
  INTERFACE_LIB_VERSION ${PROJECT_VERSION}
  INTERFACE_COMPILE_FEATURES cxx_std_14)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/campConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
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
install(FILES
  "${PROJECT_BINARY_DIR}/include/camp/config.hpp"
  DESTINATION
  include/camp)

if(CAMP_ENABLE_TESTS)
enable_testing()
add_subdirectory(test)
endif()

