cmake_minimum_required (VERSION 3.12)

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
target_include_directories (camp INTERFACE include)
set_target_properties (camp PROPERTIES
  INTERFACE_LIB_VERSION $camp_VERSION
  INTERFACE_COMPILE_FEATURES cxx_std_14)

enable_testing()
add_subdirectory(test)
