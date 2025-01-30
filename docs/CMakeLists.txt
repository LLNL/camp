###############################################################################
# Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
# and Camp project contributors. See the camp/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (CAMP_ENABLE_DOCUMENTATION)
  if (SPHINX_FOUND)
    message(STATUS "Found Sphinx")
    blt_add_sphinx_target( camp_docs )
  endif()

  if (DOXYGEN_FOUND)
    add_subdirectory(doxygen)
  endif()

  if ( NOT SPHINX_FOUND AND NOT DOXGEN_FOUND)
    message(WARNING "CAMP_ENABLE_DOCUMENTATION=On, but Sphinx or Doxygen not found. \
      Documentation won't be built.")
  endif ()
endif()
