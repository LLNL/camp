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

#include <camp/camp.hpp>
#include <stdexcept>
#include <string>
#include <exception>


namespace camp
{

void throw_re(const char *s) { throw std::runtime_error(s); }

#ifdef CAMP_ENABLE_CUDA


cudaError_t cudaAssert(cudaError_t code,
                              const char *call,
                              const char *file,
                              int line)
{
  if (code != cudaSuccess && code != cudaErrorNotReady) {
    std::string msg;
    msg += "campCudaErrchk(";
    msg += call;
    msg += ") ";
    msg += cudaGetErrorString(code);
    msg += " ";
    msg += file;
    msg += ":";
    msg += std::to_string(line);
    throw std::runtime_error(msg);
  }
  return code;
}

#endif  //#ifdef CAMP_ENABLE_CUDA

#ifdef CAMP_ENABLE_HIP

hipError_t hipAssert(hipError_t code,
                            const char *call,
                            const char *file,
                            int line)
{
  if (code != hipSuccess && code != hipErrorNotReady) {
    std::string msg;
    msg += "campHipErrchk(";
    msg += call;
    msg += ") ";
    msg += hipGetErrorString(code);
    msg += " ";
    msg += file;
    msg += ":";
    msg += std::to_string(line);
    throw std::runtime_error(msg);
  }
  return code;
}

#endif  //#ifdef CAMP_ENABLE_HIP

}  // namespace camp
