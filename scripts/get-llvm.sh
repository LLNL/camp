#!/bin/bash
set -eux

VER=$1
curl -O https://apt.llvm.org/llvm.sh
if [[ "${VER}" = 'latest' ]] ; then
    VER=all
fi
bash llvm.sh ${VER} all

ln -s /usr/bin/clang-${VER} /usr/bin/clang
ln -s /usr/bin/clang++-${VER} /usr/bin/clang++
