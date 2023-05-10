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

#include "camp/resource.hpp"

#include "camp/camp.hpp"
#include "gtest/gtest.h"

using namespace camp::resources;

// compatible but different resource for conversion test
struct Host2 : Host {
};

TEST(CampResource, Construct) { Resource h1{Host()}; }
TEST(CampResource, Copy)
{
  Resource h1{Host()};
  auto h2 = h1;
  Resource h3 = h1;
}
TEST(CampResource, ConvertFails)
{
  Resource h1{Host()};
  h1.get<Host>();
  ASSERT_THROW(h1.get<Host2>(), std::runtime_error);
  ASSERT_FALSE(h1.try_get<Host2>());
}
TEST(CampResource, GetPlatform)
{
  ASSERT_EQ(Resource(Host()).get_platform(), Platform::host);
#ifdef CAMP_HAVE_CUDA
  ASSERT_EQ(Resource(Cuda()).get_platform(), Platform::cuda);
#endif
#ifdef CAMP_HAVE_HIP
  ASSERT_EQ(Resource(Hip()).get_platform(), Platform::hip);
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  ASSERT_EQ(Resource(Omp()).get_platform(), Platform::omp_target);
#endif
}
TEST(CampResource, ConvertWorks)
{
  Resource h1{Host()};
  ASSERT_TRUE(h1.try_get<Host>());
  ASSERT_EQ(h1.get<Host>().get_platform(), Platform::host);
}

#if defined(CAMP_HAVE_CUDA)
TEST(CampResource, Reassignment)
{
  Resource h1{Host()};
  Resource c1{Cuda()};
  h1 = Cuda();
  ASSERT_EQ(typeid(c1), typeid(h1));

  Resource h2{Host()};
  Resource c2{Cuda()};
  c2 = Host();
  ASSERT_EQ(typeid(c2), typeid(h2));
}

TEST(CampResource, MultipleDevices)
{
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);

  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);

  for (int d = 0; d < num_devices; ++d) {
    cudaSetDevice(d);

    Cuda c1{Cuda::CudaFromStream(0)};
    Cuda c2{Cuda::CudaFromStream(0, d)};
    Cuda c3{};
    Cuda c4{0};
    Cuda c5{0, d};

    EXPECT_EQ(c1.get_device(), d);
    EXPECT_EQ(c2.get_device(), d);
    EXPECT_EQ(c3.get_device(), d);
    EXPECT_EQ(c4.get_device(), d);
    EXPECT_EQ(c5.get_device(), d);

    EXPECT_EQ(c1.get_stream(), cudaStream_t{0});
    EXPECT_EQ(c2.get_stream(), cudaStream_t{0});
    EXPECT_EQ(c4.get_stream(), c5.get_stream());

    const int N = 5;
    int* d_array1 = c1.allocate<int>(N);
    c1.deallocate(d_array1);
  }

  cudaSetDevice(cur_dev);
}

TEST(CampResource, DifferentDevice)
{
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);

  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);

  if (num_devices > 1) {
    int diff_dev = (cur_dev + 1) % num_devices;

    Cuda c1{Cuda::CudaFromStream(0, diff_dev)};
    Cuda c2{0, diff_dev};

    EXPECT_EQ(c1.get_device(), diff_dev);
    EXPECT_EQ(c2.get_device(), diff_dev);

    const int N = 5;
    int* d_array1 = c1.allocate<int>(N);
    c1.deallocate(d_array1);
  }

  int check_dev = -1;
  cudaGetDevice(&check_dev);

  EXPECT_EQ(check_dev, cur_dev);
}

TEST(CampResource, StreamSelect)
{
  cudaStream_t stream1, stream2;

  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  Resource c1{Cuda::CudaFromStream(stream1)};
  Resource c2{Cuda::CudaFromStream(stream2)};

  const int N = 5;
  int* d_array1 = c1.allocate<int>(N);
  int* d_array2 = c2.allocate<int>(N);

  c1.deallocate(d_array1);
  c2.deallocate(d_array2);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}

TEST(CampResource, Get)
{
  Resource dev_host{Host()};
  Resource dev_cuda{Cuda()};

  auto erased_host = dev_host.get<Host>();
  Host pure_host;
  ASSERT_EQ(typeid(erased_host), typeid(pure_host));

  auto erased_cuda = dev_cuda.get<Cuda>();
  Cuda pure_cuda;
  ASSERT_EQ(typeid(erased_cuda), typeid(pure_cuda));
}

TEST(CampResource, GetEvent)
{
  Resource h1{Host()};
  Resource c1{Cuda()};

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
  Resource h1{Host()};
  Resource c1{Cuda()};

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
#endif
#if defined(CAMP_HAVE_HIP)
TEST(CampResource, Reassignment)
{
  Resource h1{Host()};
  Resource c1{Hip()};
  h1 = Hip();
  ASSERT_EQ(typeid(c1), typeid(h1));

  Resource h2{Host()};
  Resource c2{Hip()};
  c2 = Host();
  ASSERT_EQ(typeid(c2), typeid(h2));
}

TEST(CampResource, MultipleDevices)
{
  int cur_dev = 0;
  hipGetDevice(&cur_dev);

  int num_devices = 0;
  hipGetDeviceCount(&num_devices);

  for (int d = 0; d < num_devices; ++d) {
    hipSetDevice(d);

    Hip c1{Hip::HipFromStream(0)};
    Hip c2{Hip::HipFromStream(0, d)};
    Hip c3{};
    Hip c4{0};
    Hip c5{0, d};

    EXPECT_EQ(c1.get_device(), d);
    EXPECT_EQ(c2.get_device(), d);
    EXPECT_EQ(c3.get_device(), d);
    EXPECT_EQ(c4.get_device(), d);
    EXPECT_EQ(c5.get_device(), d);

    EXPECT_EQ(c1.get_stream(), hipStream_t{0});
    EXPECT_EQ(c2.get_stream(), hipStream_t{0});
    EXPECT_EQ(c4.get_stream(), c5.get_stream());

    const int N = 5;
    int* d_array1 = c1.allocate<int>(N);
    c1.deallocate(d_array1);
  }

  hipSetDevice(cur_dev);
}

TEST(CampResource, DifferentDevice)
{
  int cur_dev = 0;
  hipGetDevice(&cur_dev);

  int num_devices = 0;
  hipGetDeviceCount(&num_devices);

  if (num_devices > 1) {
    int diff_dev = (cur_dev + 1) % num_devices;

    Hip c1{Hip::HipFromStream(0, diff_dev)};
    Hip c2{0, diff_dev};

    EXPECT_EQ(c1.get_device(), diff_dev);
    EXPECT_EQ(c2.get_device(), diff_dev);

    const int N = 5;
    int* d_array1 = c1.allocate<int>(N);
    c1.deallocate(d_array1);
  }

  int check_dev = -1;
  hipGetDevice(&check_dev);

  EXPECT_EQ(check_dev, cur_dev);
}

TEST(CampResource, StreamSelect)
{
  hipStream_t stream1, stream2;

  hipStreamCreate(&stream1);
  hipStreamCreate(&stream2);

  Resource c1{Hip::HipFromStream(stream1)};
  Resource c2{Hip::HipFromStream(stream2)};

  const int N = 5;
  int* d_array1 = c1.allocate<int>(N);
  int* d_array2 = c2.allocate<int>(N);

  c1.deallocate(d_array1);
  c2.deallocate(d_array2);

  hipStreamDestroy(stream1);
  hipStreamDestroy(stream2);
}

TEST(CampResource, Get)
{
  Resource dev_host{Host()};
  Resource dev_cuda{Hip()};

  auto erased_host = dev_host.get<Host>();
  Host pure_host;
  ASSERT_EQ(typeid(erased_host), typeid(pure_host));

  auto erased_cuda = dev_cuda.get<Hip>();
  Hip pure_cuda;
  ASSERT_EQ(typeid(erased_cuda), typeid(pure_cuda));
}

TEST(CampResource, GetEvent)
{
  Resource h1{Host()};
  Resource c1{Hip()};

  auto ev1 = h1.get_event();
  Event evh{HostEvent()};
  ASSERT_EQ(typeid(evh), typeid(ev1));

  auto ev2 = c1.get_event();
  hipStream_t s;
  hipStreamCreate(&s);
  Event evc{HipEvent(s)};
  ASSERT_EQ(typeid(evc), typeid(ev2));
}

TEST(CampEvent, Get)
{
  Resource h1{Host()};
  Resource c1{Hip()};

  Event erased_host_event = h1.get_event();
  Event erased_cuda_event = c1.get_event();

  auto pure_host_event = erased_host_event.get<HostEvent>();
  auto pure_cuda_event = erased_cuda_event.get<HipEvent>();

  HostEvent host_event;
  hipStream_t s;
  hipStreamCreate(&s);
  HipEvent cuda_event(s);

  ASSERT_EQ(typeid(host_event), typeid(pure_host_event));
  ASSERT_EQ(typeid(cuda_event), typeid(pure_cuda_event));
}
#endif

template<typename Res>
static EventProxy<Res> do_stuff(Res r)
{
  return EventProxy<Res>(r);
}

TEST(CampEventProxy, Get)
{
  Host h1{Host{}};

  {
    EventProxy<Resource> ep{Resource{h1}};
    Event e = ep;
  }

  {
    EventProxy<Host> ep{h1};
    Event e = ep;
  }

  {
    EventProxy<Host> ep{h1};
    HostEvent e = ep;
    CAMP_ALLOW_UNUSED_LOCAL(e);
  }

  {
    Event e = do_stuff(Resource{h1});
  }

  {
    Event e = do_stuff(h1);
  }

  {
    HostEvent e = do_stuff(h1);
    CAMP_ALLOW_UNUSED_LOCAL(e);
  }

  {
    do_stuff(h1);
  }

  {
    EventProxy<Resource> ep{Resource{h1}};
    Event e = ep.get();
  }

  {
    EventProxy<Host> ep{h1};
    Event e = ep.get();
  }

  {
    EventProxy<Host> ep{h1};
    HostEvent e = ep.get();
    CAMP_ALLOW_UNUSED_LOCAL(e);
  }
}

TEST(CampResource, Wait) {
  auto h = camp::resources::Host();
  h.wait();
  Event he = h.get_event_erased();
  h.wait_for(&he);
  Resource r(h);
  r.wait();
}
