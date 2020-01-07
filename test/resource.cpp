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
#include "camp/resource.hpp"

using namespace camp::resources;

TEST(CampResource, Reassignment)
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

TEST(CampResource, GetPlatform)
{
  Context dev_host{Host()};
  Context dev_cuda{Cuda()};

  ASSERT_EQ(dev_host.get_platform(), Platform::host);
  ASSERT_EQ(dev_cuda.get_platform(), Platform::cuda);
}

TEST(CampResource, Get)
{
  Context dev_host{Host()};
  Context dev_cuda{Cuda()};

  auto erased_host = dev_host.get<Host>();
  Host pure_host;
  ASSERT_EQ(typeid(erased_host), typeid(pure_host));

  auto erased_cuda = dev_cuda.get<Cuda>();
  Cuda pure_cuda;
  ASSERT_EQ(typeid(erased_cuda), typeid(pure_cuda));
}

TEST(CampResource, GetEvent)
{
  Context h1{Host()};
  Context c1{Cuda()};

  auto ev1 = h1.get_event();
  Event evh{HostEvent()};
  ASSERT_EQ(typeid(evh), typeid(ev1));

  auto ev2 = c1.get_event();
  cudaStream_t s;
  cudaStreamCreate(&s);
  Event evc{CudaEvent(s)};
  ASSERT_EQ(typeid(evc), typeid(ev2));
}

TEST(CampEvent, Get)
{
  Context h1{Host()};
  Context c1{Cuda()};

  Event erased_host_event = h1.get_event();
  Event erased_cuda_event = c1.get_event();

  auto pure_host_event = erased_host_event.get<HostEvent>();
  auto pure_cuda_event = erased_cuda_event.get<CudaEvent>();

  HostEvent host_event;
  cudaStream_t s;
  cudaStreamCreate(&s);
  CudaEvent cuda_event(s);

  ASSERT_EQ(typeid(host_event), typeid(pure_host_event));
  ASSERT_EQ(typeid(cuda_event), typeid(pure_cuda_event));
}
