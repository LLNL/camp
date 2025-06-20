//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __CAMP_SYCL_HPP
#define __CAMP_SYCL_HPP

#include "camp/config.hpp"

#ifdef CAMP_ENABLE_SYCL

#include "camp/defines.hpp"
#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#include <sycl/sycl.hpp>

#include <map>
#include <array>
#include <mutex>

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    class SyclEvent
    {
    public:
      // TODO: make this actually work
      SyclEvent(sycl::queue *CAMP_UNUSED_ARG(qu)) { m_event = sycl::event(); }
      bool check() const { return true; }
      void wait() const { getSyclEvent_t().wait(); }
      sycl::event getSyclEvent_t() const { return m_event; }

    private:
      sycl::event m_event;
    };

    class Sycl
    {
      static sycl::queue *get_a_queue(sycl::context &syclContext,
                                      int num,
                                      bool useContext)
      {
        constexpr auto gpuSelector = sycl::gpu_selector_v;
        static sycl::property_list propertyList =
            sycl::property_list(sycl::property::queue::in_order());
        static sycl::context privateContext;
        static sycl::context *contextInUse = NULL;
        static std::map<sycl::context *, std::array<sycl::queue, 16>> queueMap;


        static std::mutex m_mtx;
        m_mtx.lock();

        // User passed a context, use it
        if (useContext) {
          contextInUse = &syclContext;

          if (queueMap.find(contextInUse) == queueMap.end()) {
            queueMap[contextInUse] = {
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList)};
          }
        } else {  // User did not pass context, use last used or private one
          if (contextInUse == NULL) {
            contextInUse = &privateContext;
            queueMap[contextInUse] = {
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList),
                sycl::queue(*contextInUse, gpuSelector, propertyList)};
          }
        }
        m_mtx.unlock();

        static int previous = 0;

        static std::once_flag m_onceFlag;
        CAMP_ALLOW_UNUSED_LOCAL(m_onceFlag);
        if (num < 0) {
          m_mtx.lock();
          previous = (previous + 1) % 16;
          m_mtx.unlock();
          return &queueMap[contextInUse][previous];
        }

        return &queueMap[contextInUse][num % 16];
      }

      // Private from-queue constructor
      Sycl(sycl::queue& q) : qu(&q) {}

    public:
      Sycl(int group = -1)
      {
        sycl::context temp;
        qu = get_a_queue(temp, group, false);
      }

      Sycl(sycl::context &syclContext, int group = -1)
          : qu(get_a_queue(syclContext, group, true))
      {
      }

      /// Create a resource from a custom queue
      static Sycl SyclFromQueue(sycl::queue& q)
      {
        return Sycl(q);
      }

      // Methods
      Platform get_platform() const { return Platform::sycl; }
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
      T *allocate(size_t size, MemoryAccess ma = MemoryAccess::Device)
      {
        T *ret = nullptr;
        if (size > 0) {
          ret = sycl::malloc_shared<T>(size, *qu);
          switch (ma) {
            case MemoryAccess::Unknown:
            case MemoryAccess::Device:
              ret = sycl::malloc_device<T>(size, *qu);
              break;
            case MemoryAccess::Pinned:
              ret = sycl::malloc_host<T>(size, *qu);
              break;
            case MemoryAccess::Managed:
              ret = sycl::malloc_shared<T>(size, *qu);
              break;
          }
        }
        return ret;
      }
      void *calloc(size_t size, MemoryAccess ma = MemoryAccess::Device)
      {
        void *p = allocate<char>(size, ma);
        this->memset(p, 0, size);
        return p;
      }
      void deallocate(void *p, MemoryAccess ma = MemoryAccess::Device)
      {
        CAMP_ALLOW_UNUSED_LOCAL(ma);
        sycl::free(p, *qu);
      }
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
      sycl::queue const *get_queue() const { return qu; }

      /*
       * \brief Compares two (Sycl) resources to see if they are equal.
       *
       * \return True or false depending on if this is the same queue.
       */
      bool operator==(Sycl const& s) const
      {
        return (get_queue() == s.get_queue());
      }
      
      /*
       * \brief Compares two (Sycl) resources to see if they are NOT equal.
       *
       * \return Negation of == operator
       */
      bool operator!=(Sycl const& s) const
      {
        return !(*this == s);
      }

    private:
      sycl::queue *qu;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_ENABLE_SYCL

#endif /* __CAMP_SYCL_HPP */
