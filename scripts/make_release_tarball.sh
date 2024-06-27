#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
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
