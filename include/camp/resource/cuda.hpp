//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __CAMP_CUDA_HPP
#define __CAMP_CUDA_HPP

#include "camp/config.hpp"

#ifdef CAMP_ENABLE_CUDA

#include "camp/defines.hpp"
#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#include <cuda_runtime.h>
#include <mutex>

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
          campCudaErrchkDiscardReturn(cudaGetDevice(&prev_device));
          if (device != prev_device) {
            campCudaErrchkDiscardReturn(cudaSetDevice(device));
          } else {
            prev_device = -1;
          }
        }

        ~device_guard()
        {
          if (prev_device != -1) {
            campCudaErrchkDiscardReturn(cudaSetDevice(prev_device));
          }
        }

        int prev_device = -1;
      };

    }  // namespace

    class CudaEvent
    {
    public:
      CudaEvent(cudaStream_t stream) { init(stream); }

      CudaEvent(Cuda &res);

      bool check() const
      {
        return (campCudaErrchk(cudaEventQuery(m_event)) == cudaSuccess);
      }
      void wait() const { campCudaErrchkDiscardReturn(cudaEventSynchronize(m_event)); }
      cudaEvent_t getCudaEvent_t() const { return m_event; }

    private:
      cudaEvent_t m_event;

      void init(cudaStream_t stream)
      {
        campCudaErrchkDiscardReturn(
            cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
        campCudaErrchkDiscardReturn(cudaEventRecord(m_event, stream));
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
              campCudaErrchkDiscardReturn(cudaStreamCreate(&s));
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
      Cuda(cudaStream_t s, int dev = 0) : stream(s), device(dev) {}

      MemoryAccess get_access_type(void *p) {
        cudaPointerAttributes a;
        cudaError_t status = cudaPointerGetAttributes(&a, p);
        if (status == cudaSuccess) {
          switch(a.type){
            case cudaMemoryTypeUnregistered:
              return MemoryAccess::Unknown;
            case cudaMemoryTypeHost:
              return MemoryAccess::Pinned;
            case cudaMemoryTypeDevice:
              return MemoryAccess::Device;
            case cudaMemoryTypeManaged:
              return MemoryAccess::Managed;
          }
        }
        ::camp::throw_re("invalid pointer detected");
        // This return statement exists because compilers do not determine the
        // above unconditionally throws
        // related: https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun
        return MemoryAccess::Unknown;
      }
    public:
      Cuda(int group = -1, int dev = 0)
          : stream(get_a_stream(group)), device(dev)
      {
      }

      /// Create a resource from a custom stream
      /// The device specified must match the stream, if none is specified the
      /// currently selected device is used.
      static Cuda CudaFromStream(cudaStream_t s, int dev = -1)
      {
        if (dev < 0) {
          campCudaErrchkDiscardReturn(cudaGetDevice(&dev));
        }
        return Cuda(s, dev);
      }

      // Methods
      Platform get_platform() const { return Platform::cuda; }
      static Cuda get_default()
      {
        static Cuda c([] {
          cudaStream_t s;
#if CAMP_USE_PLATFORM_DEFAULT_STREAM
          s = 0;
#else
          campCudaErrchkDiscardReturn(cudaStreamCreate(&s));
#endif
          return s;
        }());
        return c;
      }

      CudaEvent get_event() { return CudaEvent(*this); }

      Event get_event_erased() { return Event{CudaEvent(*this)}; }

      void wait()
      {
        auto d{device_guard(device)};
        campCudaErrchkDiscardReturn(cudaStreamSynchronize(stream));
      }

      void wait_for(Event *e)
      {
        auto *cuda_event = e->try_get<CudaEvent>();
        if (cuda_event) {
          auto d{device_guard(device)};
          campCudaErrchkDiscardReturn(cudaStreamWaitEvent(get_stream(),
                                             cuda_event->getCudaEvent_t(),
                                             0));
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
          auto d{device_guard(device)};
          switch (ma) {
            case MemoryAccess::Unknown:
            case MemoryAccess::Device:
              campCudaErrchkDiscardReturn(cudaMalloc(&ret, sizeof(T) * size));
              break;
            case MemoryAccess::Pinned:
              // TODO: do a test here for whether managed is *actually* shared
              // so we can use the better performing memory
              campCudaErrchkDiscardReturn(cudaMallocHost(&ret, sizeof(T) * size));
              break;
            case MemoryAccess::Managed:
              campCudaErrchkDiscardReturn(cudaMallocManaged(&ret, sizeof(T) * size));
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
      void deallocate(void *p, MemoryAccess ma = MemoryAccess::Unknown)
      {
        auto d{device_guard(device)};
        if(ma == MemoryAccess::Unknown) {
          ma = get_access_type(p);
        }
        switch (ma) {
          case MemoryAccess::Device:
            campCudaErrchkDiscardReturn(cudaFree(p));
            break;
          case MemoryAccess::Pinned:
            // TODO: do a test here for whether managed is *actually* shared
            // so we can use the better performing memory
            campCudaErrchkDiscardReturn(cudaFreeHost(p));
            break;
          case MemoryAccess::Managed:
            campCudaErrchkDiscardReturn(cudaFree(p));
            break;
          case MemoryAccess::Unknown:
            ::camp::throw_re("Unknown memory access type, cannot free");
        }
      }
      void memcpy(void *dst, const void *src, size_t size)
      {
        if (size > 0) {
          auto d{device_guard(device)};
          campCudaErrchkDiscardReturn(
              cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
        }
      }
      void memset(void *p, int val, size_t size)
      {
        if (size > 0) {
          auto d{device_guard(device)};
          campCudaErrchkDiscardReturn(cudaMemsetAsync(p, val, size, stream));
        }
      }

      cudaStream_t get_stream() const { return stream; }
      int get_device() const { return device; }

      /*
       * \brief Compares two (Cuda) resources to see if they are equal.
       *
       * \return True or false depending on if it is the same stream.
       */
      bool operator==(Cuda const& c) const
      {
        return (get_stream() == c.get_stream());
      }
      
      /*
       * \brief Compares two (Cuda) resources to see if they are NOT equal.
       *
       * \return Negation of == operator
       */
      bool operator!=(Cuda const& c) const
      {
        return !(*this == c);
      }

    private:
      cudaStream_t stream;
      int device;
    };

    inline CudaEvent::CudaEvent(Cuda &res)
    {
      auto d{device_guard(res.get_device())};
      init(res.get_stream());
    }

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_ENABLE_CUDA

#endif /* __CAMP_CUDA_HPP */
