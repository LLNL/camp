cmake_minimum_required (VERSION 3.10)

project (camp
  LANGUAGES CXX
  VERSION 0.1.0)

option(ENABLE_CUDA "whether to build with cuda" Off)

if(ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  enable_language(CUDA)
endif()

add_library (camp INTERFACE)
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

enable_testing()
add_subdirectory(test)

