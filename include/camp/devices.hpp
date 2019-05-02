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


using default_device = cuda_t;
using default_device = cpu_t;

namespace camp {
namespace devices {

enum class Launch { undefined, sync, async };

enum class Platform { undefined = 0, host = 1, cuda = 2, omp_target = 4, hcc = 8 };

class Device {
  static Device get_default() {
  }

  static Device get_default_sync();

  static Device get(int num=0, int sync_queue_id=0);

  static void sync_all();

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

  virtual void sync_with(Device d, bool nowait=false) = 0;

  virtual void *get_sync_id() = 0;

  virtual void * alloc(size_t) = 0;

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

#if defined(_OPENMP)
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
class CudaDevice final : public Device {
  static CudaDevice& get_default()
  {
    cudaStreamCreate(&m_stream);
    static CudaDevice* dev = new CudaDevice(true, 0, sync_id);
    return dev;
  }

  static CudaDevice& get_default_sync()
  {
    static CudaDevice* dev = new CudaDevice(false, 0, nullptr);
    return dev;
  }

  static CudaDevice& get(int num=0, void* sync_queue_id=nullptr)
  {
    static CudaDevice dev = new CudaDevice(true, num, sync_queue_id);
    return dev;
  }

  bool async() override
  {
    return m_async;
  }

  static void sync_all() override
  {
    int device_count;
    cudaGetDeviceCount(&device_count);

    int previous_device;
    cudaGetDevice(&previous_device);

    for (int i = 0; i < dev_count; ++i) {
      cudaSetDevice(i)
      cudaDeviceSynchronize();
    }

    cudaSetDevice(previous_device);
  }

  void sync(bool all_queues=false) override
  {
    if (all_queues) {
      cudaDeviceSynchronize();
    } else {
      cudaStreamSynchronize(m_stream);
    }
  }

  void sync_with(Device d, bool nowait=false)
  {
    if (d.getPlatform() == Platform::cuda) {
      cudaEvent_t event;
      cudaEventCreate(&event);

      cudaEventRecord(event, d->get_sync_id());
      cudaStreamWaitEvent(m_stream, event);
    } else {
      d.sync();
    }
  }

  void* get_sync_id() override {
    return &m_sync;
  }

private:
  CudaDevice(bool async, int device_num, cudaStream_t stream) :
    Device(Platform::cuda, async ? Launch::async : Launch::sync),
    m_stream(stream),
    m_sync(async)
  {
    if (stream)
  }

  cudaStream_t m_stream;
  bool m_sync
};
#endif

} // end of namespace devices
} // end of namespace camp

#endif /* __CAMP_DEVICES_HPP */
