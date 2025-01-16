###############################################################################
# Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
# and Camp project contributors. See the camp/LICENSE file for details.
#
###############################################################################
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#  See the LLVM_LICENSE file at http://github.com/llnl/camp for the full license
#  text.
###############################################################################

find_program(SPHINX_EXECUTABLE
        NAMES sphinx-build sphinx-build2
        DOC "Path to sphinx-build executable")

# Handle REQUIRED and QUIET arguments
# this will also set SPHINX_FOUND to true if SPHINX_EXECUTABLE exists
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx
        "Failed to locate sphinx-build executable"
        SPHINX_EXECUTABLE)
