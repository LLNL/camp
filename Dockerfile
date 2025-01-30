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
ARG CMAKE_BUILD_OPTS="--build build --verbose --parallel ${PARALLEL} ${BUILD_EXTRA}"
ARG CUDA_IMG_SUFFIX="-devel-ubuntu20.04"

### start compiler base images ###
# there is no official container in the hub, but there is an official script
# to install clang/llvm by version, installs a bit more than we need, but we
# do not have to maintain it, so I'm alright with that
FROM ghcr.io/rse-ops/clang-ubuntu-20.04:llvm-${VER} AS clang
ENV LD_LIBRARY_PATH=/opt/view/lib

FROM gcc:${VER} AS gcc

FROM nvidia/cuda:${VER}${CUDA_IMG_SUFFIX} AS nvcc

FROM nvcr.io/nvidia/nvhpc:21.9-devel-cuda11.4-ubuntu20.04 AS nvhpc

FROM rocm/dev-ubuntu-20.04:${VER} AS rocm

# The intel-runtime container no longer works, use the fat one
FROM intel/oneapi:${VER} AS oneapi
RUN bash -c 'echo . /opt/intel/oneapi/setvars.sh >> ~/setup_env.sh'
### end compiler base images ###

FROM ${BASE_IMG} AS base
ARG VER
ARG BASE_IMG
ENV GTEST_COLOR=1
COPY --chown=axom:axom ./scripts/ /scripts/
WORKDIR /home/
RUN /scripts/get-deps.sh
# This is duplicative, but allows us to cache the dep installation while
# changing the sources and scripts, saves time in the development loop
COPY --chown=axom:axom . /home/

FROM base AS test
ARG PRE_CMD
ARG CTEST_OPTIONS
ARG CMAKE_OPTIONS
ARG CMAKE_BUILD_OPTS
ARG COMPILER
ENV COMPILER=${COMPILER:-g++}
RUN /bin/bash -c "[[ -f ~/setup_env.sh ]] && source ~/setup_env.sh ; ${PRE_CMD} && cmake ${CMAKE_OPTIONS} -DCMAKE_CXX_COMPILER=${COMPILER} ."
RUN /bin/bash -c "[[ -f ~/setup_env.sh ]] && source ~/setup_env.sh ; ${PRE_CMD} && cmake ${CMAKE_BUILD_OPTS}"
RUN /bin/bash -c "[[ -f ~/setup_env.sh ]] && source ~/setup_env.sh ; ${PRE_CMD} && cd build && ctest ${CTEST_OPTIONS}"

# this is here to stop azure from downloading oneapi for every test
FROM alpine AS download_fast
