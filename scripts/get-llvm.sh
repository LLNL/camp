#!/bin/bash

###############################################################################
# Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
# and Camp project contributors. See the camp/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set -eux

VER=$1
curl -O https://apt.llvm.org/llvm.sh
if [[ "${VER}" = 'latest' ]] ; then
    VER=all
fi
bash llvm.sh ${VER} all

ln -s /usr/bin/clang-${VER} /usr/bin/clang
ln -s /usr/bin/clang++-${VER} /usr/bin/clang++
