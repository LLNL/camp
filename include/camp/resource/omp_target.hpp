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
      static Omp &get_default()
      {
        static Omp o;
        return o;
      }
      OmpEvent get_event() { return OmpEvent(addr, dev); }
      Event get_event_erased() { return get_event(); }
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
        int hdev = omp_get_initial_device();
        T *hostmem = new T[size];
        register_ptr_host(hostmem, hdev);

        T *devmem = static_cast<T *>(omp_target_alloc(sizeof(T) * size, dev));
        register_ptr_dev(devmem, dev);

        register_host_devptr(hostmem, devmem);

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
#pragma omp critical(camp_register_ptr)
        {
          get_dev_register().erase(p);
        }
        omp_target_free(p, dev);
      }
      void memcpy(void *dst, const void *src, size_t size)
      {
        // this is truly, insanely awful, need to think of something better
        //int dd = get_ptr_dev(dst);
        //int sd = get_ptr_dev(src);
        // extra cast due to GCC openmp header bug
        //omp_target_memcpy(dst, (void *)src, size, 0, 0, dd, sd);

        // witness the sadness . . .
        // check for src = host, dst = device
        auto ith = get_host_register().find(src);
        auto itd = get_dev_register().find(dst);
        if (ith != get_host_register().end() && itd != get_dev_register().end())
        {
          int dd = get_ptr_dev(dst);
          int sd = get_ptr_host(src);
          // extra cast due to GCC openmp header bug
          omp_target_memcpy(dst, (void *)src, size, 0, 0, dd, sd);
        }
        // check for src = device, dst = host
        else
        {
          auto itth = get_host_register().find(dst);
          auto ittd = get_dev_register().find(src);
          if (itth != get_host_register().end() && ittd != get_dev_register().end())
          {
            int dd = get_ptr_dev(src);
            int sd = get_ptr_host(dst);
            // extra cast due to GCC openmp header bug
            omp_target_memcpy(dst, (void *)src, size, 0, 0, dd, sd);
          }
          else
          {
            printf( "TROUBLE TROUBLE TROUBLE TROUBLE TROUBLE\n" );
          }
        }
      }

      void memset(void *p, int val, size_t size)
      {
        char *local_addr = addr;
        CAMP_ALLOW_UNUSED_LOCAL(local_addr);
        char *pc = (char *)p;
#pragma omp target teams distribute parallel for device(dev) \
    depend(inout                                             \
           : local_addr[0]) is_device_ptr(pc) nowait
        for (size_t i = 0; i < size; ++i) {
          pc[i] = val;
        }
      }

      void register_ptr_host(void *p, int device)
      {
        get_host_register()[p] = device;
      }
      int get_ptr_host(void const *p)
      {
        int ret = omp_get_initial_device();
        auto it = get_host_register().find(p);
        if (it != get_host_register().end())
        {
          ret = it->second;
        }
        return ret;
      }

      void register_host_devptr(void *h, void *d)
      {
        get_host_devptr()[h] = d;
      }
      const void * obtain_devptr_withhost(void *h)
      {
        const void * ret = nullptr;
        auto it = get_host_devptr().find(h);
        if (it != get_host_devptr().end())
        {
          ret = it->second;
        }

        if (ret != nullptr)
        {
          return ret;
        }
        else
        {
          //fflush();
          abort();
          return nullptr;
        }
      }

      void register_ptr_dev(void *p, int device)
      {
#pragma omp critical(camp_register_ptr)
        {
          get_dev_register()[p] = device;
        }
      }
      int get_ptr_dev(void const *p)
      {
        int ret = omp_get_initial_device();
#pragma omp critical(camp_register_ptr)
        {
          auto it = get_dev_register().find(p);
          if (it != get_dev_register().end()) {
            ret = it->second;
          }
        }
        return ret;
      }

    private:
      char *addr;
      int dev;
      template <typename always_void_odr_helper = void>
      std::map<const void *, int> &get_dev_register()
      {
        // key: device ptr, value: device num
        static std::map<const void *, int> dev_register;
        return dev_register;
      }

      template <typename always_void_odr_helper = void>
      std::map<const void *, int> &get_host_register()
      {
        // key: host ptr, value: device num
        static std::map<const void *, int> host_register;
        return host_register;
      }

      template <typename always_void_odr_helper = void>
      std::map<const void *, const void*> &get_host_devptr()
      {
        // key: host ptr, value: device ptr
        static std::map<const void *, const void *> host_devptr;
        return host_devptr;
      }
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_HAVE_OMP_OFFLOAD

#endif /* __CAMP_OMP_TARGET_HPP */
