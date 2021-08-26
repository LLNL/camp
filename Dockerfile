###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

ARG BASE_IMG=gcc
ARG COMPILER=g++
ARG VER=latest
ARG PRE_CMD="true"
ARG BUILD_TYPE=RelWithDebInfo
ARG CTEST_EXTRA="-E '(.*offload|blt.*smoke)'"
ARG CTEST_OPTIONS="${CTEST_EXTRA} -T test -V "
ARG CMAKE_EXTRA=""
ARG CMAKE_OPTIONS="-G Ninja -B build ${CMAKE_EXTRA} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DENABLE_WARNINGS=On"
ARG PARALLEL=4
ARG BUILD_EXTRA=""
ARG CMAKE_BUILD_OPTS="--build build --parallel ${PARALLEL} ${BUILD_EXTRA}"
ARG CUDA_IMG_SUFFIX="-devel-ubuntu18.04"

FROM ubuntu:bionic AS clang_base
RUN apt-get update && apt-get install -y --no-install-recommends gpg gpg-agent wget curl software-properties-common unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

### start compiler base images ###
# there is no official container in the hub, but there is an official script
# to install clang/llvm by version, installs a bit more than we need, but we
# do not have to maintain it, so I'm alright with that
FROM clang_base AS clang
ARG VER
ADD ./scripts/get-llvm.sh get-llvm.sh
RUN ./get-llvm.sh $VER bah

FROM gcc:${VER} AS gcc

FROM nvidia/cuda:${VER}${CUDA_IMG_SUFFIX} AS nvcc

FROM rocm/dev-ubuntu-20.04:${VER} AS rocm

# use the runtime container and then have it install the compiler,
# save us a few gigabytes every time
FROM intel/oneapi-runtime:${VER} AS oneapi
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        intel-oneapi-compiler-dpcpp-cpp && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# inject all setvars stuff into environment file
RUN bash -c 'echo . /opt/intel/oneapi/setvars.sh >> ~/setup_env.sh'
### end compiler base images ###

FROM ${BASE_IMG} AS base
ARG VER
ARG BASE_IMG
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/
WORKDIR /home/
RUN ./scripts/get-deps.sh

FROM base AS test
ARG PRE_CMD
ARG CTEST_OPTIONS
ARG CMAKE_OPTIONS
ARG CMAKE_BUILD_OPTS
ARG COMPILER
ENV COMPILER=${COMPILER:-g++}
ENV HCC_AMDGPU_TARGET=gfx900
RUN /bin/bash -c "[[ -f ~/setup_env.sh ]] && source ~/setup_env.sh ; ${PRE_CMD} && cmake ${CMAKE_OPTIONS} -DCMAKE_CXX_COMPILER=${COMPILER} ."
RUN /bin/bash -c "[[ -f ~/setup_env.sh ]] && source ~/setup_env.sh ; ${PRE_CMD} && cmake ${CMAKE_BUILD_OPTS}"
RUN /bin/bash -c "[[ -f ~/setup_env.sh ]] && source ~/setup_env.sh ; ${PRE_CMD} && cd build && ctest ${CTEST_OPTIONS}"

# this is here to stop azure from downloading oneapi for every test
FROM alpine AS download_fast
