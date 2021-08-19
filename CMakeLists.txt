cmake_minimum_required (VERSION 3.9)

project (camp
  LANGUAGES CXX
  VERSION 0.1.0)

# Default to C++11 if not set so GTest/GMock can build
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++14" CACHE STRING "")
endif()

include(cmake/load_blt.cmake)


cmake_dependent_option(CAMP_ENABLE_TESTS "Build tests" On "ENABLE_TESTS" Off)


# if ENABLE_TESTS is defined by a parent project, and
# CAMP_ENABLE_TESTS has not been set to OFF set the
# value of CAMP_ENABLE_TESTS to the value of ENABLE_TESTS.
if (CAMP_ENABLE_TESTS AND DEFINED ENABLE_TESTS)
  set(CAMP_ENABLE_TESTS ${ENABLE_TESTS})
endif()

# generate list of headers at configure time, no this is not perfect
# note that this *DOES* work on windows if bash is installed, which
# it is on azure
execute_process(COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/gen-header-list.sh
  OUTPUT_FILE ${PROJECT_BINARY_DIR}/camp_headers.cmake
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
execute_process(COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/gen-header-list.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
include(${PROJECT_BINARY_DIR}/camp_headers.cmake)

set (camp_depends)

if (ENABLE_OPENMP)
  set (camp_depends
    openmp)
endif()

if(ENABLE_CUDA)
  if("${CUDA_VERSION_STRING}" VERSION_LESS "9.2")
    message(FATAL_ERROR "Trying to use CUDA version ${CUDA_VERSION_STRING}. CAMP dependency Googletest requires CUDA version 9.2.x or newer.")
  endif()
endif()

if(ENABLE_HIP)
  if("${HIP_VERSION_STRING}" VERSION_LESS "3.5")
    message(FATAL_ERROR "Trying to use HIP/ROCm version ${HIP_VERSION_STRING}. CAMP requires HIP/ROCm version 3.5 or newer. ")
  endif()
endif()

if (ENABLE_CUDA)
  set(camp_depends
    ${camp_depends}
    cuda)
  if(ENABLE_NV_TOOLS_EXT)
    set(camp_depends
      ${camp_depends}
      nvtoolsext)
  endif ()
endif ()

if (ENABLE_EXTERNAL_CUB)
  set(camp_depends
    ${camp_depends}
    cub)
endif ()

if (ENABLE_HIP)
  set(camp_depends
    ${camp_depends}
    hip)
endif ()
blt_add_library (
  NAME camp
  HEADERS ${camp_headers}
  DEPENDS_ON ${camp_depends}
  )
target_include_directories (camp INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )
set_target_properties (camp PROPERTIES
  INTERFACE_LIB_VERSION $camp_VERSION
  INTERFACE_COMPILE_FEATURES cxx_std_11)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/campConfigVersion.cmake"
  VERSION 0.1.0
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

