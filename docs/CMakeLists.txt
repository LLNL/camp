###############################################################################
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Maintained by Tom Scogland <scogland1@llnl.gov>
# CODE-756261, All rights reserved.
# This file is part of camp.
# For details about use and distribution, please read LICENSE and NOTICE from
# http://github.com/llnl/camp
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
