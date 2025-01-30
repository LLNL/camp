/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

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
