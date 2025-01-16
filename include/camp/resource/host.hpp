//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//See the LLVM_LICENSE file at http://github.com/llnl/camp for the full license
//text.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __CAMP_HOST_HPP
#define __CAMP_HOST_HPP

#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#include <cstdlib>
#include <cstring>

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    class HostEvent
    {
    public:
      HostEvent() {}
      bool check() const { return true; }
      void wait() const {}
    };

    class Host
    {
    public:
      Host(int /* group */ = -1) {}

      // Methods
      Platform get_platform() const { return Platform::host; }
      static Host get_default()
      {
        static Host h;
        return h;
      }
      HostEvent get_event() { return HostEvent(); }
      Event get_event_erased()
      {
        Event e{HostEvent()};
        return e;
      }
      void wait() {}
      void wait_for(Event *e) { e->wait(); }

      // Memory
      template <typename T>
      T *allocate(size_t n, MemoryAccess = MemoryAccess::Device)
      {
        return (T *)std::malloc(sizeof(T) * n);
      }
      void *calloc(size_t size, MemoryAccess = MemoryAccess::Device)
      {
        void *p = allocate<char>(size);
        this->memset(p, 0, size);
        return p;
      }
      void deallocate(void *p, MemoryAccess = MemoryAccess::Device) { std::free(p); }
      void memcpy(void *dst, const void *src, size_t size) { std::memcpy(dst, src, size); }
      void memset(void *p, int val, size_t size) { std::memset(p, val, size); }

      /*
       * \brief Compares two (Host) resources to see if they are equal.
       *
       * \return Always return true since we are on the Host in this case.
       */
      bool operator==(Host const&) const
      {
        return true;
      }
      
      /*
       * \brief Compares two (Host) resources to see if they are NOT equal.
       *
       * \return Always return false. Host resources are always the same.
       */
      bool operator!=(Host const&) const
      {
        return false;
      }
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif /* __CAMP_DEVICES_HPP */
