/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_OMP_TARGET_HPP
#define __CAMP_OMP_TARGET_HPP

#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#ifdef CAMP_HAVE_OMP_OFFLOAD
#include <omp.h>

#include <map>
#include <memory>

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    struct hdmem
    {
      void * host_arr[10];
    };

    class OmpEvent
    {
    public:
      OmpEvent(char *addr_in, int device = omp_get_default_device())
          : addr(addr_in), dev(device)
      {
#pragma omp target device(dev) depend(inout : addr_in[0]) nowait
        {
        }
      }
      bool check() const
      {
        // think up a way to do something better portably
        return false;
      }
      void wait() const
      {
        char *local_addr = addr;
        CAMP_ALLOW_UNUSED_LOCAL(local_addr);
        // if only we could use taskwait depend portably...
#pragma omp task if (0) depend(inout : local_addr[0])
        {
        }
      }
      void *getEventAddr() const { return addr; }

    private:
      char *addr;
      int dev;
    };

    class Omp
    {
      static char *get_addr(int num)
      {
        static char addrs[16] = {};
        static int previous = 0;

        static std::mutex m_mtx;

        if (num < 0) {
          m_mtx.lock();
          previous = (previous + 1) % 16;
          m_mtx.unlock();
          return &addrs[previous];
        }

        return &addrs[num % 16];
      }

    public:
      Omp(int group = -1, int device = omp_get_default_device())
          : addr(get_addr(group)), dev(device)
      {
      }

      // Methods
      Platform get_platform() { return Platform::omp_target; }
      static Omp get_default()
      {
        static Omp o;
        return o;
      }
      OmpEvent get_event() { return OmpEvent(addr, dev); }
      Event get_event_erased() { return Event{get_event()}; }
      void wait()
      {
        char *local_addr = addr;
        CAMP_ALLOW_UNUSED_LOCAL(local_addr);
#pragma omp target device(dev) depend(inout : local_addr[0])
        {
        }
      }
      void wait_for(Event *e)
      {
        OmpEvent *oe = e->try_get<OmpEvent>();
        if (oe) {
          char *local_addr = addr;
          char *other_addr = (char *)oe->getEventAddr();
          CAMP_ALLOW_UNUSED_LOCAL(local_addr);
          CAMP_ALLOW_UNUSED_LOCAL(other_addr);
#pragma omp target depend(inout                      \
                          : local_addr[0]) depend(in \
                                                  : other_addr[0]) nowait
          {
          }
        } else {
          e->wait();
        }
      }

      // Memory
      template <typename T>
      T *allocate(size_t size)
      {
        T * hostmem = (T*)malloc(sizeof(T)*sizehost);

#pragma omp target enter data map( to: hostmem[0:size] )

        return hostmem;
      }

      void *calloc(size_t size)
      {
        void *p = allocate<char>(size);
        this->memset(p, 0, size);
        return p;
      }

      void deallocate(void *p)
      {
        char * pp = (char *)p;
        CAMP_ALLOW_UNUSED_LOCAL(pp);
#pragma omp target exit data map( release: pp[:0] )
        free(p);
      }

      void memcpy(void *dst, const void *src, size_t size)
      {
        int initdev = omp_get_initial_device();
        int sdevice = omp_target_is_present( (void *)src, dev ) ? dev : initdev;
        int ddevice = omp_target_is_present( (void *)dst, dev ) ? dev : initdev;
        #pragma omp target data if(sdevice != initdev) device(sdevice) use_device_ptr(src)
        #pragma omp target data if(ddevice != initdev) device(ddevice) use_device_ptr(dst)
        {
          omp_target_memcpy(dst, (void *)src, size, 0, 0, ddevice, sdevice);
        }
      }

      void memset(void *p, int val, size_t size)
      {
        char *local_addr = addr;
        CAMP_ALLOW_UNUSED_LOCAL(local_addr);
        char *pc = (char *)p;
#pragma omp target data use_device_ptr(pc)
#pragma omp target teams distribute parallel for device(dev) \
    depend(inout                                             \
           : local_addr[0]) is_device_ptr(pc) nowait
        for (size_t i = 0; i < size; ++i) {
          pc[i] = val;
        }
      }

    private:
      char *addr;
      int dev;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_HAVE_OMP_OFFLOAD

#endif /* __CAMP_OMP_TARGET_HPP */
