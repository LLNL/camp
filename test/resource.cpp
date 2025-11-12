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

struct Host2 : Host {
};

struct NotAResource {
};

static_assert(is_host_resource<Host>::value,
              "Host should satisfy is_host_resource");
static_assert(is_host_resource<Host&>::value,
              "Host& should satisfy is_host_resource");
static_assert(is_host_resource<const Host>::value,
              "const Host should satisfy is_host_resource");
static_assert(is_host_resource<const Host&>::value,
              "const Host& should satisfy is_host_resource");

#ifdef CAMP_HAVE_CUDA
static_assert(!is_host_resource<Cuda>::value,
              "Cuda should not satisfy is_host_resource");
#endif
#ifdef CAMP_HAVE_HIP
static_assert(!is_host_resource<Hip>::value,
              "Hip should not satisfy is_host_resource");
#endif
#ifdef CAMP_HAVE_SYCL
static_assert(!is_host_resource<Sycl>::value,
              "Sycl should not satisfy is_host_resource");
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
static_assert(!is_host_resource<Omp>::value,
              "Omp should not satisfy is_host_resource");
#endif

static_assert(!is_host_resource<int>::value,
              "int should not satisfy is_host_resource");
static_assert(!is_host_resource<NotAResource>::value,
              "NotAResource should not satisfy is_host_resource");
static_assert(!is_host_resource<void*>::value,
              "void* should not satisfy is_host_resource");
static_assert(!is_host_resource<Host2>::value,
              "Host2 (derived type) should not satisfy is_host_resource");

#ifdef CAMP_HAVE_CUDA
static_assert(is_cuda_resource<Cuda>::value,
              "Cuda should satisfy is_cuda_resource");
static_assert(is_cuda_resource<Cuda&>::value,
              "Cuda& should satisfy is_cuda_resource");
static_assert(is_cuda_resource<const Cuda>::value,
              "const Cuda should satisfy is_cuda_resource");

static_assert(!is_cuda_resource<Host>::value,
              "Host should not satisfy is_cuda_resource");
#ifdef CAMP_HAVE_HIP
static_assert(!is_cuda_resource<Hip>::value,
              "Hip should not satisfy is_cuda_resource");
#endif

static_assert(!is_cuda_resource<int>::value,
              "int should not satisfy is_cuda_resource");
static_assert(!is_cuda_resource<NotAResource>::value,
              "NotAResource should not satisfy is_cuda_resource");
#endif

#ifdef CAMP_HAVE_HIP
static_assert(is_hip_resource<Hip>::value,
              "Hip should satisfy is_hip_resource");
static_assert(is_hip_resource<Hip&>::value,
              "Hip& should satisfy is_hip_resource");
static_assert(is_hip_resource<const Hip>::value,
              "const Hip should satisfy is_hip_resource");

static_assert(!is_hip_resource<Host>::value,
              "Host should not satisfy is_hip_resource");
#ifdef CAMP_HAVE_CUDA
static_assert(!is_hip_resource<Cuda>::value,
              "Cuda should not satisfy is_hip_resource");
#endif

static_assert(!is_hip_resource<int>::value,
              "int should not satisfy is_hip_resource");
static_assert(!is_hip_resource<NotAResource>::value,
              "NotAResource should not satisfy is_hip_resource");
#endif

#ifdef CAMP_HAVE_SYCL
static_assert(is_sycl_resource<Sycl>::value,
              "Sycl should satisfy is_sycl_resource");
static_assert(is_sycl_resource<Sycl&>::value,
              "Sycl& should satisfy is_sycl_resource");
static_assert(is_sycl_resource<const Sycl>::value,
              "const Sycl should satisfy is_sycl_resource");

static_assert(!is_sycl_resource<Host>::value,
              "Host should not satisfy is_sycl_resource");

static_assert(!is_sycl_resource<int>::value,
              "int should not satisfy is_sycl_resource");
static_assert(!is_sycl_resource<NotAResource>::value,
              "NotAResource should not satisfy is_sycl_resource");
#endif

#ifdef CAMP_HAVE_OMP_OFFLOAD
static_assert(is_omp_resource<Omp>::value,
              "Omp should satisfy is_omp_resource");
static_assert(is_omp_resource<Omp&>::value,
              "Omp& should satisfy is_omp_resource");
static_assert(is_omp_resource<const Omp>::value,
              "const Omp should satisfy is_omp_resource");

static_assert(!is_omp_resource<Host>::value,
              "Host should not satisfy is_omp_resource");

static_assert(!is_omp_resource<int>::value,
              "int should not satisfy is_omp_resource");
static_assert(!is_omp_resource<NotAResource>::value,
              "NotAResource should not satisfy is_omp_resource");
#endif

static_assert(is_resource<Host>::value,
              "Host should satisfy is_resource");
static_assert(is_resource<Host&>::value,
              "Host& should satisfy is_resource");
static_assert(is_resource<const Host>::value,
              "const Host should satisfy is_resource");
static_assert(is_resource<const Host&>::value,
              "const Host& should satisfy is_resource");
static_assert(is_resource<Host&&>::value,
              "Host&& should satisfy is_resource");

#ifdef CAMP_HAVE_CUDA
static_assert(is_resource<Cuda>::value,
              "Cuda should satisfy is_resource");
static_assert(is_resource<Cuda&>::value,
              "Cuda& should satisfy is_resource");
static_assert(is_resource<const Cuda>::value,
              "const Cuda should satisfy is_resource");
#endif

#ifdef CAMP_HAVE_HIP
static_assert(is_resource<Hip>::value,
              "Hip should satisfy is_resource");
static_assert(is_resource<Hip&>::value,
              "Hip& should satisfy is_resource");
static_assert(is_resource<const Hip>::value,
              "const Hip should satisfy is_resource");
#endif

#ifdef CAMP_HAVE_SYCL
static_assert(is_resource<Sycl>::value,
              "Sycl should satisfy is_resource");
static_assert(is_resource<Sycl&>::value,
              "Sycl& should satisfy is_resource");
static_assert(is_resource<const Sycl>::value,
              "const Sycl should satisfy is_resource");
#endif

#ifdef CAMP_HAVE_OMP_OFFLOAD
static_assert(is_resource<Omp>::value,
              "Omp should satisfy is_resource");
static_assert(is_resource<Omp&>::value,
              "Omp& should satisfy is_resource");
static_assert(is_resource<const Omp>::value,
              "const Omp should satisfy is_resource");
#endif

static_assert(!is_resource<int>::value,
              "int should not satisfy is_resource");
static_assert(!is_resource<float>::value,
              "float should not satisfy is_resource");
static_assert(!is_resource<double>::value,
              "double should not satisfy is_resource");
static_assert(!is_resource<void*>::value,
              "void* should not satisfy is_resource");
static_assert(!is_resource<char*>::value,
              "char* should not satisfy is_resource");
static_assert(!is_resource<NotAResource>::value,
              "NotAResource should not satisfy is_resource");
static_assert(!is_resource<Host2>::value,
              "Host2 (derived type) should not satisfy is_resource");

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
  ASSERT_EQ(static_cast<const Resource>(Host()).get_platform(), Platform::host);
#ifdef CAMP_HAVE_CUDA
  ASSERT_EQ(static_cast<const Resource>(Cuda()).get_platform(), Platform::cuda);
#endif
#ifdef CAMP_HAVE_HIP
  ASSERT_EQ(static_cast<const Resource>(Hip()).get_platform(), Platform::hip);
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  ASSERT_EQ(static_cast<const Resource>(Omp()).get_platform(), Platform::omp_target);
#endif
}
TEST(CampResource, Compare)
{
  Resource h1{Host()};
  Resource h2{Host()};
  Host h; Resource h3{h};

#ifdef CAMP_HAVE_CUDA
  Resource r1{Cuda()};
  Resource r2{Cuda()};
  Cuda s; Resource r3{s};
#endif
#ifdef CAMP_HAVE_HIP
  Resource r1{Hip()};
  Resource r2{Hip()};
  Hip s; Resource r3{s};
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  Resource r1{Omp()};
  Resource r2{Omp()};
  Omp s; Resource r3{s};
#endif

  ASSERT_TRUE(h1 == h1);
  ASSERT_TRUE(h1 == h2);
  
  ASSERT_FALSE(h1 != h2);

#if defined(CAMP_HAVE_CUDA) || \
    defined(CAMP_HAVE_HIP) || \
    defined(CAMP_HAVE_OMP_OFFLOAD)
  ASSERT_TRUE(r1 == r1);
  ASSERT_TRUE(r2 == r2);
  ASSERT_TRUE(s == s);
  ASSERT_TRUE(h == h);
  ASSERT_TRUE(r1 != r2);
  ASSERT_TRUE(r2 != r1);
  ASSERT_TRUE(r1 != h1);
  ASSERT_TRUE(r3 != h3);

  ASSERT_FALSE(r1 == r2);
  ASSERT_FALSE(r2 == r1);
  ASSERT_FALSE(r2 == r3);
  ASSERT_FALSE(h2 == r2);
  ASSERT_FALSE(h3 == r3);
  ASSERT_FALSE(r1 != r1);
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

TEST(CampResourceTypeTraits, HelperTraits) {
  // Test is_host_resource
  ASSERT_TRUE(is_host_resource<Host>::value);
  ASSERT_TRUE(is_host_resource<Host&>::value);
  ASSERT_TRUE(is_host_resource<const Host>::value);
  ASSERT_FALSE(is_host_resource<int>::value);
  ASSERT_FALSE(is_host_resource<NotAResource>::value);
  ASSERT_FALSE(is_host_resource<Host2>::value);

#ifdef CAMP_HAVE_CUDA
  // Test is_cuda_resource
  ASSERT_TRUE(is_cuda_resource<Cuda>::value);
  ASSERT_TRUE(is_cuda_resource<Cuda&>::value);
  ASSERT_TRUE(is_cuda_resource<const Cuda>::value);
  ASSERT_FALSE(is_cuda_resource<Host>::value);
  ASSERT_FALSE(is_cuda_resource<int>::value);
  ASSERT_FALSE(is_cuda_resource<NotAResource>::value);

  // Test cross-backend exclusivity
  ASSERT_FALSE(is_host_resource<Cuda>::value);
  ASSERT_FALSE(is_cuda_resource<Host>::value);
#endif

#ifdef CAMP_HAVE_HIP
  // Test is_hip_resource
  ASSERT_TRUE(is_hip_resource<Hip>::value);
  ASSERT_TRUE(is_hip_resource<Hip&>::value);
  ASSERT_TRUE(is_hip_resource<const Hip>::value);
  ASSERT_FALSE(is_hip_resource<Host>::value);
  ASSERT_FALSE(is_hip_resource<int>::value);
  ASSERT_FALSE(is_hip_resource<NotAResource>::value);
#endif

#ifdef CAMP_HAVE_SYCL
  // Test is_sycl_resource
  ASSERT_TRUE(is_sycl_resource<Sycl>::value);
  ASSERT_TRUE(is_sycl_resource<Sycl&>::value);
  ASSERT_TRUE(is_sycl_resource<const Sycl>::value);
  ASSERT_FALSE(is_sycl_resource<Host>::value);
  ASSERT_FALSE(is_sycl_resource<int>::value);
  ASSERT_FALSE(is_sycl_resource<NotAResource>::value);
#endif

#ifdef CAMP_HAVE_OMP_OFFLOAD
  // Test is_omp_resource
  ASSERT_TRUE(is_omp_resource<Omp>::value);
  ASSERT_TRUE(is_omp_resource<Omp&>::value);
  ASSERT_TRUE(is_omp_resource<const Omp>::value);
  ASSERT_FALSE(is_omp_resource<Host>::value);
  ASSERT_FALSE(is_omp_resource<int>::value);
  ASSERT_FALSE(is_omp_resource<NotAResource>::value);
#endif
}

TEST(CampResourceTypeTraits, IsResourceTrait) {
  // Test is_resource with valid resource types
  ASSERT_TRUE(is_resource<Host>::value);
  ASSERT_TRUE(is_resource<Host&>::value);
  ASSERT_TRUE(is_resource<const Host>::value);
  ASSERT_TRUE(is_resource<const Host&>::value);

#ifdef CAMP_HAVE_CUDA
  ASSERT_TRUE(is_resource<Cuda>::value);
  ASSERT_TRUE(is_resource<Cuda&>::value);
  ASSERT_TRUE(is_resource<const Cuda>::value);
#endif

#ifdef CAMP_HAVE_HIP
  ASSERT_TRUE(is_resource<Hip>::value);
  ASSERT_TRUE(is_resource<Hip&>::value);
  ASSERT_TRUE(is_resource<const Hip>::value);
#endif

#ifdef CAMP_HAVE_SYCL
  ASSERT_TRUE(is_resource<Sycl>::value);
  ASSERT_TRUE(is_resource<Sycl&>::value);
  ASSERT_TRUE(is_resource<const Sycl>::value);
#endif

#ifdef CAMP_HAVE_OMP_OFFLOAD
  ASSERT_TRUE(is_resource<Omp>::value);
  ASSERT_TRUE(is_resource<Omp&>::value);
  ASSERT_TRUE(is_resource<const Omp>::value);
#endif

  // Test is_resource with non-resource types
  ASSERT_FALSE(is_resource<int>::value);
  ASSERT_FALSE(is_resource<float>::value);
  ASSERT_FALSE(is_resource<double>::value);
  ASSERT_FALSE(is_resource<void*>::value);
  ASSERT_FALSE(is_resource<char*>::value);
  ASSERT_FALSE(is_resource<NotAResource>::value);
  ASSERT_FALSE(is_resource<Host2>::value);
}
