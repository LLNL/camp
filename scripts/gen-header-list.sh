#!/bin/sh

###############################################################################
# Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
# and Camp project contributors. See the camp/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

echo "set(camp_headers"
find include -name '*.hpp' | grep -v '\.in\.hpp'
echo ")"
