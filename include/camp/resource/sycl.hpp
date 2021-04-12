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
      SyclEvent(sycl::queue qu)
      {
        m_event = sycl::event();
      }
      bool check() const { return true; }
      void wait() const { getSyclEvent_t().wait(); }
      sycl::event getSyclEvent_t() const { return m_event; }

    private:
      sycl::event m_event;
    };

    class Sycl
    {
      static sycl::queue get_a_stream(int num)
      {
        static sycl::queue streams[16];
        static int previous = 0;

        static std::once_flag m_onceFlag;
        static std::mutex m_mtx;

        if (num < 0) {
          m_mtx.lock();
          previous = (previous + 1) % 16;
          m_mtx.unlock();
          return streams[previous];
        }

        return streams[num % 16];
      }

    public:
      Sycl(int group = -1) : stream(get_a_stream(group)) {}

      // Methods
      Platform get_platform() { return Platform::sycl; }
      static Sycl &get_default()
      {
        static Sycl h;
        return h;
      }
      SyclEvent get_event() { return SyclEvent(get_stream()); }
      Event get_event_erased() { return Event{SyclEvent(get_stream())}; }
      void wait() { stream.wait(); }
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
          ret = sycl::malloc_shared<T>(size, stream); 
        }
        return ret;
      }
      void *calloc(size_t size)
      {
        void *p = allocate<char>(size);
        this->memset(p, 0, size);
        return p;
      }
      void deallocate(void *p)
      { 
        sycl::free(p, stream);
      }
      void memcpy(void *dst, const void *src, size_t size)
      {
        if (size > 0) {
          stream.memcpy(dst, src, size).wait();
        }
      }
      void memset(void *p, int val, size_t size)
      {
        if (size > 0) {
          stream.memset(p, val, size);
          stream.wait();
        }
      }

      sycl::queue get_stream() { return stream; }

    private:
      sycl::queue stream;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_HAVE_SYCL

#endif /* __CAMP_SYCL_HPP */
