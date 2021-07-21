/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_HIP_HPP
#define __CAMP_HIP_HPP

#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#ifdef CAMP_HAVE_HIP
#include <hip/hip_runtime.h>

namespace camp
{
namespace resources
{
  inline namespace v1
  {
    class Hip;

    namespace
    {
      struct device_guard {
        device_guard(int device)
        {
          campHipErrchk(hipGetDevice(&prev_device));
          campHipErrchk(hipSetDevice(device));
        }

        ~device_guard() { campHipErrchk(hipSetDevice(prev_device)); }

      int prev_device;
    };

    }  // namespace
    class HipEvent
    {
    public:
      HipEvent(hipStream_t stream) { init(stream); }

      HipEvent(Hip& res);

      bool check() const { return (campHipErrchk(hipEventQuery(m_event)) == hipSuccess); }
      void wait() const { campHipErrchk(hipEventSynchronize(m_event)); }
      hipEvent_t getHipEvent_t() const { return m_event; }

    private:
      hipEvent_t m_event;

      void init(hipStream_t stream)
      {
        campHipErrchk(hipEventCreateWithFlags(&m_event, hipEventDisableTiming));
        campHipErrchk(hipEventRecord(m_event, stream));
      }
    };

    class Hip
    {
      static hipStream_t get_a_stream(int num)
      {
        static hipStream_t streams[16] = {};
        static int previous = 0;

        static std::once_flag m_onceFlag;
        static std::mutex m_mtx;

        std::call_once(m_onceFlag, [] {
          if (streams[0] == nullptr) {
            for (auto &s : streams) {
              campHipErrchk(hipStreamCreate(&s));
            }
          }
        });

        if (num < 0) {
          m_mtx.lock();
          previous = (previous + 1) % 16;
          m_mtx.unlock();
          return streams[previous];
        }

        return streams[num % 16];
      }

      // Private from-stream constructor
      Hip(hipStream_t s, int dev=0) : stream(s), device(dev) {}

    public:
      Hip(int group = -1, int dev=0) : stream(get_a_stream(group)), device(dev) {}

      /// Create a resource from a custom stream
      /// The device specified must match the stream, if none is specified the
      /// currently selected device is used.
      static Hip HipFromStream(hipStream_t s, int dev=-1) {
        if (dev < 0) {
          campHipErrchk(hipGetDevice(&dev));
        }
        return Hip(s, dev);
      }

      // Methods
      Platform get_platform() { return Platform::hip; }
      static Hip get_default()
      {
        static Hip h( [] {
          hipStream_t s;
#if CAMP_USE_PLATFORM_DEFAULT_STREAM
          s = 0;
#else
          campHipErrchk(hipStreamCreate(&s));
#endif
          return s;
        }());
        return h;
      }

      HipEvent get_event()
      {
        return HipEvent(*this);
      }

      Event get_event_erased()
      {
        return Event{HipEvent(*this)};
      }

      void wait()
      {
        auto d{device_guard(device)};
        campHipErrchk(hipStreamSynchronize(stream));
      }

      void wait_for(Event *e)
      {
        auto *hip_event = e->try_get<HipEvent>();
        if (hip_event) {
          auto d{device_guard(device)};
          campHipErrchk(hipStreamWaitEvent(get_stream(),
                                           hip_event->getHipEvent_t(),
                                           0));
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
          auto d{device_guard(device)};
          campHipErrchk(hipMallocManaged(&ret, sizeof(T) * size));
        }
        return ret;
      }
      void *calloc(size_t size)
      {
        void *p = allocate<char>(size);
        this->memset(p, 0, size);
        return p;
      }
      void deallocate(void *p) {
        auto d{device_guard(device)};
        campHipErrchk(hipFree(p));
      }
      void memcpy(void *dst, const void *src, size_t size)
      {
        if (size > 0) {
          auto d{device_guard(device)};
          campHipErrchk(hipMemcpyAsync(dst, src, size, hipMemcpyDefault, stream));
        }
      }
      void memset(void *p, int val, size_t size)
      {
        if (size > 0) {
          auto d{device_guard(device)};
          campHipErrchk(hipMemsetAsync(p, val, size, stream));
        }
      }

      hipStream_t get_stream() { return stream; }
      int get_device() { return device; }

    private:
      hipStream_t stream;
      int device;
    };

    inline HipEvent::HipEvent(Hip& res)
    {
      auto d{device_guard(res.get_device())};
      init(res.get_stream());
    }

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_HAVE_HIP

#endif /* __CAMP_HIP_HPP */
