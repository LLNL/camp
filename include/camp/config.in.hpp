//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//See the LLVM_LICENSE file at http://github.com/llnl/camp for the full license
//text.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Define CAMP_CONFIG_OVERRIDE to change this on a per-file basis
#if !defined(CAMP_CONFIG_OVERRIDE)
#cmakedefine CAMP_ENABLE_OPENMP
#cmakedefine CAMP_ENABLE_TARGET_OPENMP
#cmakedefine CAMP_ENABLE_CUDA
#cmakedefine CAMP_ENABLE_HIP
#cmakedefine CAMP_ENABLE_SYCL
#cmakedefine CAMP_WIN_STATIC_BUILD
#endif

#define CAMP_VERSION_MAJOR @camp_VERSION_MAJOR@
#define CAMP_VERSION_MINOR @camp_VERSION_MINOR@
#define CAMP_VERSION_PATCH @camp_VERSION_PATCH@

#define CAMP_VERSION (CAMP_VERSION_MAJOR * 1000000) \
                    +(CAMP_VERSION_MINOR * 1000) \
                    +(CAMP_VERSION_PATCH)

#if (defined(_WIN32) || defined(_WIN64)) && !defined(CAMP_WIN_STATIC_BUILD)
#ifdef CAMP_DLL_EXPORTS
#define CAMP_DLL_EXPORT __declspec(dllexport)
#else
#define CAMP_DLL_EXPORT __declspec(dllimport)
#endif
#else
#define CAMP_DLL_EXPORT
#endif
