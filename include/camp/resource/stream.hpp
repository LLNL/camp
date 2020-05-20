/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_STREAM_HPP
#define __CAMP_STREAM_HPP


#if defined(CAMP_HAVE_CUDA) || defined(CAMP_HAVE_HIP)
namespace camp
{
namespace resources
{
  inline namespace v1
  {

    constexpr int DEFAULT_STREAM = -1;
    constexpr int NEXT_STREAM = -2;

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif //#if defined(CAMP_HAVE_CUDA) || defined(CAMP_HAVE_HIP)

#endif /* __CAMP_STREAM_HPP */
