#!/bin/bash
set -eux

VER=$1
curl -O https://apt.llvm.org/llvm.sh
if [[ "${VER}" = 'latest' ]] ; then
    VER=$(grep LLVM_VERSION llvm.sh | head -n 1 | sed -e 's/LLVM_VERSION=//')
fi
bash llvm.sh ${VER}

ln -s /usr/bin/clang-${VER} /usr/bin/clang
ln -s /usr/bin/clang++-${VER} /usr/bin/clang++
