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

#ifdef CAMP_ENABLE_SYCL
#include <sycl/sycl.hpp>
#include <map>
#include <array>

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
      bool check() const {
        return (m_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete);
      }
      void wait() const { getSyclEvent_t().wait(); }
      sycl::event getSyclEvent_t() const { return m_event; }

    private:
      sycl::event m_event;
    };

    class Sycl
    {
      static sycl::queue *get_a_queue(sycl::context *syclContext,
                                      int num,
                                      bool useContext)
      {
        static sycl::device gpuSelector { sycl::gpu_selector_v };
        static sycl::property_list propertyList =
            sycl::property_list(sycl::property::queue::in_order());
        static std::map<sycl::context *, std::array<sycl::queue, 16>> queueMap;


        static std::mutex m_mtx;
        m_mtx.lock();

        // User passed a context, use it
        if (useContext) {
          if (queueMap.find(contextInUse) == queueMap.end()) {
            queueMap[syclContext] = {
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
                sycl::queue(*syclContext, gpuSelector, propertyList)};
          }
        } else {  // User did not pass context, use last used or private one
          if (syclContext == nullptr) {
            sycl::context* privateContext = new sycl::context(gpuSelector);
            queueMap[privateContext] = {
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList),
                sycl::queue(*privateContext, gpuSelector, propertyList)};
          }
        }
        m_mtx.unlock();

        static int previous = 0;

        static std::once_flag m_onceFlag;
        if (num < 0) {
          m_mtx.lock();
          previous = (previous + 1) % 16;
          m_mtx.unlock();
          return &queueMap[contextInUse][previous];
        }

        return &queueMap[contextInUse][num % 16];
      }
    public:
      Sycl(int group = -1)
      {
        qu = get_a_queue(nullptr, group, false);
      }

      Sycl(sycl::context &syclContext, int group = -1)
          : qu(get_a_queue(&syclContext, group, true))
      {
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
      void wait() {
        #if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
        qu->ext_oneapi_submit_barrier();
        #else
        qu->wait();
        #endif
      }
      void wait_for(Event *e)
      {
        auto *sycl_event = e->try_get<SyclEvent>();
        if (sycl_event) {
        #if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
          qu->ext_oneapi_submit_barrier( {sycl_event->getSyclEvent_t()} );
        #else
          (sycl_event->getSyclEvent_t()).wait();
        #endif
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
      void deallocate(void *p, MemoryAccess ma = MemoryAccess::Device) { sycl::free(p, *qu); }
      void memcpy(void *dst, const void *src, size_t size)
      {
        if (size > 0) {
          qu->memcpy(dst, src, size);
        }
      }
      void memset(void *p, int val, size_t size)
      {
        if (size > 0) {
          qu->memset(p, val, size);
        }
      }

      sycl::queue *get_queue() { return qu; }

    private:
      sycl::queue *qu;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_ENABLE_SYCL

#endif /* __CAMP_SYCL_HPP */
