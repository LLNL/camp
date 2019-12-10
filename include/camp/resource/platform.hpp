/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_PLATFORM_HPP
#define __CAMP_PLATFORM_HPP

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    enum class Platform {
      undefined = 0,
      host = 1,
      omp = 2,
      tbb = 4,
      cuda = 8,
      hip = 16
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif /* __CAMP_PLATFORM_HPP */
