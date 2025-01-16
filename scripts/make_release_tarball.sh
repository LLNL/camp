#!/usr/bin/env bash

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

TAR_CMD=`which tar`
VERSION=`git describe --tags`

git archive --prefix=camp-${VERSION}/ -o camp-${VERSION}.tar HEAD 2> /dev/null

echo "Running git archive submodules..."

p=`pwd` && (echo .; git submodule foreach --recursive) | while read entering path; do
    temp="${path%\'}";
    temp="${temp#\'}";
    path=$temp;
    [ "$path" = "" ] && continue;
    (cd $path && git archive --prefix=camp-${VERSION}/$path/ HEAD > $p/tmp.tar && ${TAR_CMD} --concatenate --file=$p/camp-${VERSION}.tar $p/tmp.tar && rm $p/tmp.tar);
done

gzip camp-${VERSION}.tar
