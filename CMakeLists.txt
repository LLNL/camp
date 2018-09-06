cmake_minimum_required (VERSION 3.12)

project (camp
  LANGUAGES CXX
  VERSION 0.1.0)

add_library (camp INTERFACE)
target_include_directories (camp INTERFACE include)
set_target_properties (camp PROPERTIES
  INTERFACE_LIB_VERSION $camp_VERSION
  INTERFACE_COMPILE_FEATURES cxx_std_14)

enable_testing()
add_subdirectory(test)
