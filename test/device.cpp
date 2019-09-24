//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "camp/camp.hpp"
#include "gtest/gtest.h"
#include "camp/device.hpp"

using namespace camp::devices;

TEST(CampDevice, Construct)
{
  Context h1{Host()};
  Context c1{Cuda()};
  h1 = Cuda();
  ASSERT_EQ(typeid(c1), typeid(h1));

  Context h2{Host()};
  Context c2{Cuda()};
  c2 = Host();
  ASSERT_EQ(typeid(c2), typeid(h2));
}


TEST(CampDevice, GetPlatform)
{
  Context dev_host{Host()};
  Context dev_cuda{Cuda()};

  ASSERT_EQ(dev_host.get_platform(), Platform::host);
  ASSERT_EQ(dev_cuda.get_platform(), Platform::cuda);
}

TEST(CampDevice, Get)
{
  Context dev_host{Host()};
  Context dev_cuda{Cuda()};

  auto h = dev_host.get<Host>();
  Host pure_host();
  //ASSERT_EQ(typeid(h), typeid(pure_host));
}

TEST(CampDevice, GetEvent)
{
  Context h1{Host()};
  Context c1{Cuda()};

  auto e1 = h1.get_event();
  Event eh{HostEvent()};
  ASSERT_EQ(typeid(eh), typeid(e1));

  auto e2 = c1.get_event();
  cudaStream_t s;
  cudaStreamCreate(&s);
  Event ec{CudaEvent(s)};
  ASSERT_EQ(typeid(ec), typeid(e2));
}

TEST(CampEvent, WaitOn)
{
}

TEST(CampEvent, Construct)
{
}

TEST(CampEvent, Check)
{
}

TEST(CampEvent, Wait)
{
}

TEST(CampEvent, Get)
{
}
