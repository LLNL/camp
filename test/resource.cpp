//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "camp/resource.hpp"

#include "camp/camp.hpp"
#include "gtest/gtest.h"

using namespace camp::resources;

// compatible but different resource for conversion test
struct Host2 : Host { };
#ifdef CAMP_HAVE_CUDA
  struct Cuda2 : Cuda { };
#endif
#ifdef CAMP_HAVE_HIP
  struct Hip2 : Hip { };
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  struct Omp2 : Omp { };
#endif
#ifdef CAMP_HAVE_SYCL
  struct Sycl2 : Sycl { };
#endif

template < typename Res >
void test_construct()
{
  Resource h1{Res()};
}
//
TEST(CampResource, Construct)
{
  test_construct<Host>();
#ifdef CAMP_HAVE_CUDA
  test_construct<Cuda>();
#endif
#ifdef CAMP_HAVE_HIP
  test_construct<Hip>();
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  test_construct<Omp>();
#endif
#ifdef CAMP_HAVE_SYCL
  test_construct<Sycl>();
#endif
}

template < typename Res >
void test_copy()
{
  Resource r1{Res()};
  auto r2 = r1;
  Resource r3 = r1;
}
//
TEST(CampResource, Copy)
{
  test_copy<Host>();
#ifdef CAMP_HAVE_CUDA
  test_copy<Cuda>();
#endif
#ifdef CAMP_HAVE_HIP
  test_copy<Hip>();
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  test_copy<Omp>();
#endif
#ifdef CAMP_HAVE_SYCL
  test_copy<Sycl>();
#endif
}

template < typename Res, typename Res2 >
void test_convert_fails()
{
  Resource r{Res()};
  r.get<Res>();
  ASSERT_THROW(r.get<Res2>(), std::runtime_error);
  ASSERT_FALSE(r.try_get<Res2>());
}
//
TEST(CampResource, ConvertFails)
{
  test_convert_fails<Host, Host2>();
#ifdef CAMP_HAVE_CUDA
  test_convert_fails<Cuda, Cuda2>();
#endif
#ifdef CAMP_HAVE_HIP
  test_convert_fails<Hip, Hip2>();
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  test_convert_fails<Omp, Omp2>();
#endif
#ifdef CAMP_HAVE_SYCL
  test_convert_fails<Sycl, Sycl2>();
#endif
}

template < typename Res >
void test_convert_works(Platform platform)
{
  Resource r{Res()};
  ASSERT_TRUE(r.try_get<Res>());
  ASSERT_EQ(r.get<Res>().get_platform(), platform);
}
//
TEST(CampResource, ConvertWorks)
{
  test_convert_works<Host>(Platform::host);
#ifdef CAMP_HAVE_CUDA
  test_convert_works<Cuda>(Platform::cuda);
#endif
#ifdef CAMP_HAVE_HIP
  test_convert_works<Hip>(Platform::hip);
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  test_convert_works<Omp>(Platform::omp_target);
#endif
#ifdef CAMP_HAVE_SYCL
  test_convert_works<Sycl>(Platform::sycl);
#endif
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
#ifdef CAMP_HAVE_SYCL
  ASSERT_EQ(static_cast<const Resource>(Sycl()).get_platform(), Platform::sycl);
#endif
}

template < typename Res >
void test_compare(Resource& h1, Resource& h2, Resource& h3)
{
  Resource r1{Res()};
  Resource r2{Res()};
  Res r; Resource r3{r};

  ASSERT_TRUE(r1 == r1);
  ASSERT_TRUE(r2 == r2);
  ASSERT_TRUE(r == r);
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
}
//
TEST(CampResource, Compare)
{
  Resource h1{Host()};
  Resource h2{Host()};
  Host h; Resource h3{h};

  ASSERT_TRUE(h1 == h1);
  ASSERT_TRUE(h1 == h2);
  ASSERT_TRUE(h == h);

  ASSERT_FALSE(h1 != h2);

#ifdef CAMP_HAVE_CUDA
  test_compare<Cuda>(h1, h2, h3);
#endif
#ifdef CAMP_HAVE_HIP
  test_compare<Hip>(h1, h2, h3);
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  test_compare<Omp>(h1, h2, h3);
#endif
#ifdef CAMP_HAVE_SYCL
  test_compare<Sycl>(h1, h2, h3);
#endif
}

template < typename Res >
void test_reassignment()
{
  Resource h1{Host()};
  Resource r1{Res()};
  h1 = Res();
  ASSERT_EQ(typeid(r1), typeid(h1));

  Resource h2{Host()};
  Resource r2{Res()};
  r2 = Host();
  ASSERT_EQ(typeid(r2), typeid(h2));
}
//
TEST(CampResource, Reassignment)
{
  test_reassignment<Host>();
#ifdef CAMP_HAVE_CUDA
  test_reassignment<Cuda>();
#endif
#ifdef CAMP_HAVE_HIP
  test_reassignment<Hip>();
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  test_reassignment<Omp>();
#endif
#ifdef CAMP_HAVE_SYCL
  test_reassignment<Sycl>();
#endif
}

void test_select_stream(Resource r1, Resource r2)
{
  const int N = 5;
  int* r_array1 = r1.allocate<int>(N);
  int* r_array2 = r2.allocate<int>(N);

  r1.deallocate(r_array1);
  r2.deallocate(r_array2);
}
//
TEST(CampResource, StreamSelect)
{
  test_select_stream(Host(), Host());
#if defined(CAMP_HAVE_CUDA)
  {
    cudaStream_t stream1, stream2;
    campCudaErrchkDiscardReturn(cudaStreamCreate(&stream1));
    campCudaErrchkDiscardReturn(cudaStreamCreate(&stream2));
    test_select_stream(Cuda::CudaFromStream(stream1),
                       Cuda::CudaFromStream(stream2));
    campCudaErrchkDiscardReturn(cudaStreamDestroy(stream1));
    campCudaErrchkDiscardReturn(cudaStreamDestroy(stream2));
  }
#endif
#if defined(CAMP_HAVE_HIP)
  {
    hipStream_t stream1, stream2;
    campHipErrchkDiscardReturn(hipStreamCreate(&stream1));
    campHipErrchkDiscardReturn(hipStreamCreate(&stream2));
    test_select_stream(Hip::HipFromStream(stream1),
                       Hip::HipFromStream(stream2));
    campHipErrchkDiscardReturn(hipStreamDestroy(stream1));
    campHipErrchkDiscardReturn(hipStreamDestroy(stream2));
  }
#endif
}

template < typename Res >
void test_get()
{
  Resource dev_res{Res()};
  auto erased_res = dev_res.get<Res>();
  Res pure_res;
  ASSERT_EQ(typeid(erased_res), typeid(pure_res));
}
//
TEST(CampResource, Get)
{
  test_get<Host>();
#ifdef CAMP_HAVE_CUDA
  test_get<Cuda>();
#endif
#ifdef CAMP_HAVE_HIP
  test_get<Hip>();
#endif
#ifdef CAMP_HAVE_OMP_OFFLOAD
  test_get<Omp>();
#endif
#ifdef CAMP_HAVE_SYCL
  test_get<Sycl>();
#endif
}

template < typename Res, typename ResEvent, typename... EventArgs >
void test_get_event(EventArgs&&... eventArgs)
{
  Resource r{Res()};
  auto erased_event = r.get_event();
  Event event{ResEvent(std::forward<EventArgs>(eventArgs)...)};
  ASSERT_EQ(typeid(event), typeid(erased_event));
}
//
TEST(CampResource, GetEvent)
{
  test_get_event<Host, HostEvent>();
#if defined(CAMP_HAVE_CUDA)
  {
    cudaStream_t s;
    campCudaErrchkDiscardReturn(cudaStreamCreate(&s));
    test_get_event<Cuda, CudaEvent>(s);
    campCudaErrchkDiscardReturn(cudaStreamDestroy(s));
  }
#endif
#if defined(CAMP_HAVE_HIP)
  {
    hipStream_t s;
    campHipErrchkDiscardReturn(hipStreamCreate(&s));
    test_get_event<Hip, HipEvent>(s);
    campHipErrchkDiscardReturn(hipStreamDestroy(s));
  }
#endif
}

#if defined(CAMP_HAVE_CUDA)
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
  campCudaErrchkDiscardReturn(cudaStreamCreate(&s));
  CudaEvent cuda_event(s);

  ASSERT_EQ(typeid(host_event), typeid(pure_host_event));
  ASSERT_EQ(typeid(cuda_event), typeid(pure_cuda_event));
}
#endif
#if defined(CAMP_HAVE_HIP)
TEST(CampEvent, Get)
{
  Resource h1{Host()};
  Resource d1{Hip()};

  Event erased_host_event = h1.get_event();
  Event erased_hip_event = d1.get_event();

  auto pure_host_event = erased_host_event.get<HostEvent>();
  auto pure_hip_event = erased_hip_event.get<HipEvent>();

  HostEvent host_event;
  hipStream_t s;
  campHipErrchkDiscardReturn(hipStreamCreate(&s));
  HipEvent hip_event(s);

  ASSERT_EQ(typeid(host_event), typeid(pure_host_event));
  ASSERT_EQ(typeid(hip_event), typeid(pure_hip_event));
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
