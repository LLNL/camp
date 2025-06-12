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
      static sycl::queue *get_a_queue(sycl::context *syclContext,
                                      int num)
      {
        static constexpr auto gpuSelector = sycl::gpu_selector_v;
        static const sycl::property_list propertyList =
            sycl::property_list(sycl::property::queue::in_order());
        static constexpr int num_queues = 16;

        static std::mutex s_mtx;

        // note that this type must not invalidate iterators when modified
        using queueMap_type = std::map<sycl::context *, std::pair<int, std::array<sycl::queue, num_queues>>>;
        static queueMap_type queueMap;
        static const typename queueMap_type::iterator queueMap_end = queueMap.end();

        // Think about possibility of inconsistent behavior with prevContextIter
        static thread_local typename queueMap_type::iterator prevContextIter = queueMap_end;

        if (syclContext) {
          if (prevContextIter != queueMap_end) {
            if (syclContext != prevContextIter->first) {
              prevContextIter = queueMap_end;
            }
          }
        } else {
          if (prevContextIter == queueMap_end) {
            static sycl::context privateContext;
            syclContext = &privateContext;
          }
        }

        if (prevContextIter == queueMap_end || num < 0) {
          std::lock_guard<std::mutex> lock(s_mtx);

          if (prevContextIter == queueMap_end) {
            auto contextIter = queueMap.find(syclContext);
            if (contextIter == queueMap_end) {

              contextIter = queueMap.emplace(syclContext, { num_queues-1, {
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList),
                  sycl::queue(*syclContext, gpuSelector, propertyList)}});
            }
            prevContextIter = contextIter;
          }

          if (num < 0) {
            int& previous = prevContextIter->second.first;
            previous = (previous + 1) % num_queues;
            return &prevContextIter->second.second[previous];
          }
        }

        return &prevContextIter->second.second[num % num_queues];
      }

      // Private from-queue constructor
      Sycl(sycl::queue& q) : qu(&q) {}

    public:
      Sycl(int group = -1)
        : qu(get_a_queue(nullptr, group))
      {
      }

      Sycl(sycl::context &syclContext, int group = -1)
          : qu(get_a_queue(&syclContext, group))
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
