###############################################################################
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Maintained by Tom Scogland <scogland1@llnl.gov>
# CODE-756261, All rights reserved.
# This file is part of camp.
# For details about use and distribution, please read LICENSE and NOTICE from
# http://github.com/llnl/camp
###############################################################################

#------------------------------------------------------------------------------
# Doxygen documentation targets
#------------------------------------------------------------------------------

blt_add_doxygen_target( camp_doxygen )

# Required directory for doxygen to generate output to
add_custom_target( build_doxygen_output_dir
                   ALL
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/html
                 )

add_dependencies( camp_doxygen build_doxygen_output_dir )

add_dependencies(camp_docs
  camp_doxygen)
