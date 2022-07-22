if (NOT BLT_LOADED)
  set(BLT_EXPORT_THIRDPARTY ON CACHE BOOL "")
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "\
      The BLT submodule is not present. \
      If in git repository run the following two commands:\n \
      git submodule init\n \
      git submodule update")
    endif ()
  endif ()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

if (NOT BLT_EXPORTED)
  set(BLT_EXPORTED On CACHE BOOL "" FORCE)
  blt_import_library(NAME          blt_stub EXPORTABLE On)
            install(TARGETS blt_stub
                    EXPORT               bltTargets)
  blt_export_tpl_targets(EXPORT bltTargets NAMESPACE blt)
  install(EXPORT bltTargets
    DESTINATION  lib/cmake/camp)
endif()
