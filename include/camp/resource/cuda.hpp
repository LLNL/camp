/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_CUDA_HPP
#define __CAMP_CUDA_HPP

#include "camp/defines.hpp"
#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#ifdef CAMP_HAVE_CUDA
#include <cuda_runtime.h>

namespace camp
{
namespace resources
{
  inline namespace v1
  {
    class Cuda;

    namespace
    {
      struct device_guard {
        device_guard(int device)
        {
          campCudaErrchk(cudaGetDevice(&prev_device));
          campCudaErrchk(cudaSetDevice(device));
        }

        ~device_guard() { campCudaErrchk(cudaSetDevice(prev_device)); }

      int prev_device;
    };

    }  // namespace

    class CudaEvent
    {
    public:
      CudaEvent(cudaStream_t stream) { init(stream); }

      CudaEvent(Cuda& res);

      bool check() const { return (campCudaErrchk(cudaEventQuery(m_event)) == cudaSuccess); }
      void wait() const { campCudaErrchk(cudaEventSynchronize(m_event)); }
      cudaEvent_t getCudaEvent_t() const { return m_event; }

    private:
      cudaEvent_t m_event;

      void init(cudaStream_t stream)
      {
        campCudaErrchk(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
        campCudaErrchk(cudaEventRecord(m_event, stream));
      }
    };

    class Cuda
    {
      static cudaStream_t get_a_stream(int num)
      {
        static cudaStream_t streams[16] = {};
        static int previous = 0;

        static std::once_flag m_onceFlag;
        static std::mutex m_mtx;

        std::call_once(m_onceFlag, [] {
          if (streams[0] == nullptr) {
            for (auto &s : streams) {
              campCudaErrchk(cudaStreamCreate(&s));
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
      Cuda(cudaStream_t s, int dev=0) : stream(s), device(dev) {}

    public:
      Cuda(int group = -1, int dev=0) : stream(get_a_stream(group)), device(dev) {}

      /// Create a resource from a custom stream
      /// The device specified must match the stream, if none is specified the
      /// currently selected device is used.
      static Cuda CudaFromStream(cudaStream_t s, int dev=-1) {
        if (dev < 0) {
          campCudaErrchk(cudaGetDevice(&dev));
        }
        return Cuda(s, dev);
      }

      // Methods
      Platform get_platform() { return Platform::cuda; }
      static Cuda get_default()
      {
        static Cuda c( [] {
          cudaStream_t s;
#if CAMP_USE_PLATFORM_DEFAULT_STREAM
          s = 0;
#else
          campCudaErrchk(cudaStreamCreate(&s));
#endif
          return s;
        }());
        return c;
      }

      CudaEvent get_event()
      {
        return CudaEvent(*this);
      }

      Event get_event_erased()
      {
        return Event{CudaEvent(*this)};
      }

      void wait()
      {
        auto d{device_guard(device)};
        campCudaErrchk(cudaStreamSynchronize(stream));
      }

      void wait_for(Event *e)
      {
        auto *cuda_event = e->try_get<CudaEvent>();
        if (cuda_event) {
          auto d{device_guard(device)};
          campCudaErrchk(cudaStreamWaitEvent(get_stream(),
                                             cuda_event->getCudaEvent_t(),
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
          campCudaErrchk(cudaMallocManaged(&ret, sizeof(T) * size));
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
        auto d{device_guard(device)};
        campCudaErrchk(cudaFree(p));
      }
      void memcpy(void *dst, const void *src, size_t size)
      {
        if (size > 0) {
          auto d{device_guard(device)};
          campCudaErrchk(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
        }
      }
      void memset(void *p, int val, size_t size)
      {
        if (size > 0) {
          auto d{device_guard(device)};
          campCudaErrchk(cudaMemsetAsync(p, val, size, stream));
        }
      }

      cudaStream_t get_stream() { return stream; }
      int get_device() { return device; }

    private:
      cudaStream_t stream;
      int device;
    };

    inline CudaEvent::CudaEvent(Cuda& res)
    {
      auto d{device_guard(res.get_device())};
      init(res.get_stream());
    }

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_HAVE_CUDA

#endif /* __CAMP_CUDA_HPP */
