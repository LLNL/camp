/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_SYCL_HPP
#define __CAMP_SYCL_HPP

#include "camp/defines.hpp"
#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#ifdef CAMP_HAVE_SYCL
#include <CL/sycl.hpp>

using namespace cl;

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    class SyclEvent
    {
    public:
      SyclEvent(sycl::queue *qu) { m_event = sycl::event(); }
      bool check() const { return true; }
      void wait() const { getSyclEvent_t().wait(); }
      sycl::event getSyclEvent_t() const { return m_event; }

    private:
      sycl::event m_event;
    };

    class Sycl
    {
<<<<<<< HEAD
      static sycl::queue *get_a_queue(int num)
=======
      static sycl::queue* get_a_queue(sycl::context syclContext, int num)
>>>>>>> b3654c7 (Update SYCL resource to use single context and optionally take sycl context on construction)
      {
        static sycl::gpu_selector gpuSelector;
        static sycl::property_list propertyList = sycl::property_list(sycl::property::queue::in_order());
        static sycl::queue qus[] = { sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList),
                                     sycl::queue(syclContext, gpuSelector, propertyList) };

        static int previous = 0;

        static std::once_flag m_onceFlag;
        static std::mutex m_mtx;

        if (num < 0) {
          m_mtx.lock();
          previous = (previous + 1) % 16;
          m_mtx.unlock();
          return &qus[previous];
        }

        return &qus[num % 16];
      }

    public:
      Sycl(sycl::context syclContext = sycl::context(), int group = -1) : qu(get_a_queue(syclContext, group)) {}

      // Methods
      Platform get_platform() { return Platform::sycl; }
      static Sycl get_default()
      {
        static Sycl h;
        return h;
      }
      SyclEvent get_event() { return SyclEvent(get_queue()); }
      Event get_event_erased() { return Event{SyclEvent(get_queue())}; }
      void wait() { qu->wait(); }
      void wait_for(Event *e)
      {
        auto *sycl_event = e->try_get<SyclEvent>();
        if (sycl_event) {
          (sycl_event->getSyclEvent_t()).wait();
        } else {
          e->wait();
        }
      }

      // Memory
      template <typename T>
      T *allocate(size_t size)
      {
        T *ret = nullptr;
        if (size > 0) {
          ret = sycl::malloc_shared<T>(size, *qu);
        }
        return ret;
      }
      void *calloc(size_t size)
      {
        void *p = allocate<char>(size);
        this->memset(p, 0, size);
        return p;
      }
      void deallocate(void *p) { sycl::free(p, *qu); }
      void memcpy(void *dst, const void *src, size_t size)
      {
        if (size > 0) {
          qu->memcpy(dst, src, size).wait();
        }
      }
      void memset(void *p, int val, size_t size)
      {
        if (size > 0) {
          qu->memset(p, val, size).wait();
        }
      }

      sycl::queue *get_queue() { return qu; }

    private:
      sycl::queue *qu;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_HAVE_SYCL

#endif /* __CAMP_SYCL_HPP */