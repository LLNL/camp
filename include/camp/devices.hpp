/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_DEVICES_HPP
#define __CAMP_DEVICES_HPP

#if defined(__NVCC__)
#include <cuda.h>
#endif

#include <cstddef>

namespace camp {
namespace devices {

enum class Launch { undefined, sync, async };

enum class Platform { undefined = 0, host = 1, cuda = 2, omp_target = 4, hcc = 8 };

class Device {
  public:
  //static Device& get_default() {
  //}

  //static Device& get_default_sync();

  //static Device& get(int num=0, int sync_queue_id=0);

  virtual void sync_all() = 0;

  Platform getPlatform()
  {
    return m_platform;
  }

  Launch getLaunch()
  {
    return m_launch;
  }

  bool async()
  {
    return (m_launch == Launch::async);

  }

  virtual void sync(bool all_queues=false) = 0;

  virtual void sync_with(Device& d, bool nowait=false) = 0;

  virtual void *get_sync_id() = 0;

  virtual void * alloc(std::size_t) = 0;

  virtual void free(void const *) = 0;

protected:
  Device(Platform platform, Launch launch) :
    m_platform(platform),
    m_launch(launch)
  {
  }

  Platform m_platform;
  Launch m_launch;
  void* sync_id;
};

#if 0 //defined(_OPENMP)
struct OMPTDevice final : Device {
  OMPTDevice(int device_num=0, int sync_queue_id=0) : Device(Platform::omp_target, Launch::async){
  }

  static OMPTDevice get_default() {
    static OMPTDevice *d = new OMPTDevice();
  }

  static OMPTDevice get_default_sync();

  static OMPTDevice get(int num=0, int sync_queue_id=0);

  static void sync_all();

  virtual bool async() override;

  virtual void sync(bool all_queues=false) override;

  virtual void sync_with(OMPTDevice d, bool nowait=false)  override;

  virtual void * alloc(size_t)  override;

  virtual void free(void const *)  override;
};
#endif

#if defined(__NVCC__)

namespace {

class CuStreamPool {
  public:
    static cudaStream_t* get()
    {
      if (!m_custream_pool) {
        m_custream_pool = new CuStreamPool();
      }

      return &m_custream_pool->getNextStream();
    }

  private:
    CuStreamPool() :
      m_current_stream{0}
    {
    }

    cudaStream_t getNextStream()
    {
      size_t stream_id = m_current_stream;

      if (!m_stream_init[stream_id]) {
        cudaStreamCreate(&(m_stream[stream_id]));
      }

      m_current_stream = (stream_id+1) % STREAM_POOL_SIZE;

      return m_stream[stream_id];
    }

    static CuStreamPool* m_custream_pool;

    static const size_t STREAM_POOL_SIZE{25};
    size_t m_current_stream;

    cudaStream_t m_stream[STREAM_POOL_SIZE];
    bool m_stream_init[STREAM_POOL_SIZE] = {false};
  };

CuStreamPool* CuStreamPool::m_custream_pool = nullptr;

} 

class CudaDevice final : public Device {
  public:
  static CudaDevice& get_default()
  {
    static CudaDevice* dev = new CudaDevice(true, 0, nullptr);
    return *dev;
  }

  static CudaDevice& get_default_sync()
  {
    static CudaDevice* dev = new CudaDevice(false, 0, nullptr);
    return *dev;
  }

  static CudaDevice& get(int num=0, void* sync_queue_id=nullptr)
  {
    if (sync_queue_id == nullptr)
      sync_queue_id = CuStreamPool::get();

    static CudaDevice* dev = new CudaDevice(true, num, sync_queue_id);
    return *dev;
  }

  void sync_all() final override
  {
    int device_count;
    cudaGetDeviceCount(&device_count);

    int previous_device;
    cudaGetDevice(&previous_device);

    for (int i = 0; i < device_count; ++i) {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    cudaSetDevice(previous_device);
  }

  void sync(bool all_queues=false) final override
  {
    if (all_queues) {
      cudaDeviceSynchronize();
    } else {
      cudaStreamSynchronize(m_stream);
    }
  }

  void sync_with(Device& d, bool nowait=false)
  {
    if (d.getPlatform() == Platform::cuda) {
      cudaEvent_t event;
      cudaEventCreate(&event);

      //cudaEventRecord(event, (cudaStream_t)(*d->get_sync_id()));
      cudaStreamWaitEvent(m_stream, event, 0);
    } else {
      d.sync();
    }
  }

  void* get_sync_id() final override {
    return &m_stream;
  }

  void * alloc(size_t size) final override
  {
    void* ret;
    cudaMalloc(&ret, size);
    return ret;
  }

  void free(void const * ptr) final override
  {
    cudaFree(&ptr);
  }

private:
  CudaDevice(bool async, int device_num, void* stream) :
    Device(Platform::cuda, async ? Launch::async : Launch::sync),
    m_stream((cudaStream_t) stream)
  {
  }

  cudaStream_t m_stream;
};
#endif

} // end of namespace devices
} // end of namespace camp

#endif /* __CAMP_DEVICES_HPP */
