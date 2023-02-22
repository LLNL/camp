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

#ifdef CAMP_ENABLE_CUDA

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
          campCudaErrchk(cudaGetDevice(&prev_device));
          if (device != prev_device) {
            campCudaErrchk(cudaSetDevice(device));
          } else {
            prev_device = -1;
          }
        }

        ~device_guard()
        {
          if (prev_device != -1) {
            campCudaErrchk(cudaSetDevice(prev_device));
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
      void wait() const { campCudaErrchk(cudaEventSynchronize(m_event)); }
      cudaEvent_t getCudaEvent_t() const { return m_event; }

    private:
      cudaEvent_t m_event;

      void init(cudaStream_t stream)
      {
        campCudaErrchk(
            cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
        campCudaErrchk(cudaEventRecord(m_event, stream));
      }
    };

    class Cuda
    {
      static int get_current_device()
      {
        int dev = -1;
        campCudaErrchk(cudaGetDevice(&dev));
        return dev;
      }

      static int get_device_from_stream(cudaStream_t stream)
      {
        int dev = -1;
        CUcontext stream_ctx;
        campCuErrchk(cuStreamGetCtx(stream, &stream_ctx));
        campCuErrchk(cuCtxPushCurrent(stream_ctx));
        campCuErrchk(cuCtxGetDevice(&dev));
        campCuErrchk(cuCtxPopCurrent(stream_ctx));
        return dev;
      }

      static cudaStream_t get_a_stream(int num, int dev)
      {
        static constexpr int num_streams = 16;
        struct Streams {
          cudaStream_t streams[num_streams] = {};
          int previous = 0;

          std::once_flag onceFlag;
          std::mutex mtx;
        };

        static std::vector<Streams> devices([] {
          int count = -1;
          campcudaErrchk(cudaGetDeviceCount(&count));
          return count;
        }());

        std::call_once(devices[dev].onceFlag, [&] {
          auto d{device_guard(dev)};
          if (devices[dev].streams[0] == nullptr) {
            for (auto &s : devices[dev].streams) {
              campcudaErrchk(cudaStreamCreate(&s));
            }
          }
        });

        if (num < 0) {
          std::lock_guard<std::mutex> guard(devices[dev].mtx);
          devices[dev].previous = (devices[dev].previous + 1) % num_streams;
          num = devices[dev].previous;
        } else {
          num = num % num_streams;
        }

        return devices[dev].streams[num];
      }

      // Private from-stream constructor
      Cuda(cudaStream_t s, int dev) : stream(s), device(dev) {}

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
      Cuda(int group = -1, int dev = get_current_device())
          : stream(get_a_stream(group, dev)), device(dev)
      {
      }

      /// Create a resource from a custom stream.
      /// The device specified must match the stream, if none is specified the
      /// device is found from the stream.
      /// Note that if dev is specified then no runtime calls are made allowing
      /// this to be called before entering main.
      static Cuda CudaFromStream(cudaStream_t s, int dev = -1)
      {
        return Cuda(s, (dev >= 0) ? dev : get_device_from_stream(s));
      }

      // Methods
      Platform get_platform() { return Platform::cuda; }
      static Cuda get_default()
      {
        static Cuda c([] {
          cudaStream_t s;
#if CAMP_USE_PLATFORM_DEFAULT_STREAM
          s = 0;
#else
          campCudaErrchk(cudaStreamCreate(&s));
#endif
          return s;
        }(), get_current_device());
        return c;
      }

      CudaEvent get_event() { return CudaEvent(*this); }

      Event get_event_erased() { return Event{CudaEvent(*this)}; }

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
      T *allocate(size_t size, MemoryAccess ma = MemoryAccess::Device)
      {
        T *ret = nullptr;
        if (size > 0) {
          auto d{device_guard(device)};
          switch (ma) {
            case MemoryAccess::Unknown:
            case MemoryAccess::Device:
              campCudaErrchk(cudaMalloc(&ret, sizeof(T) * size));
              break;
            case MemoryAccess::Pinned:
              // TODO: do a test here for whether managed is *actually* shared
              // so we can use the better performing memory
              campCudaErrchk(cudaMallocHost(&ret, sizeof(T) * size));
              break;
            case MemoryAccess::Managed:
              campCudaErrchk(cudaMallocManaged(&ret, sizeof(T) * size));
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
        if (ma == MemoryAccess::Unknown) {
          ma = get_access_type(p);
        }
        switch (ma) {
          case MemoryAccess::Device:
            campCudaErrchk(cudaFree(p));
            break;
          case MemoryAccess::Pinned:
            // TODO: do a test here for whether managed is *actually* shared
            // so we can use the better performing memory
            campCudaErrchk(cudaFreeHost(p));
            break;
          case MemoryAccess::Managed:
            campCudaErrchk(cudaFree(p));
            break;
          case MemoryAccess::Unknown:
            ::camp::throw_re("Unknown memory access type, cannot free");
            break;
        }
      }
      void memcpy(void *dst, const void *src, size_t size)
      {
        if (size > 0) {
          auto d{device_guard(device)};
          campCudaErrchk(
              cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
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
