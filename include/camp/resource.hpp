/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_RESOURCE_HPP
#define __CAMP_RESOURCE_HPP

#include <cstring>
#include <memory>
#include <mutex>

#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"
#include "camp/resource/host.hpp"
#include "camp/resource/cuda.hpp"
#include "camp/resource/hip.hpp"

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    class Context
    {
    public:
      template <typename T>
      Context(T &&value)
      {
        m_value.reset(new ContextModel<T>(value));
      }
      template <typename T>
      bool test_get()
      {
        auto result = dynamic_cast<ContextModel<T> *>(m_value.get());
        return (result != nullptr);
      }
      template <typename T>
      T get()
      {
        auto result = dynamic_cast<ContextModel<T> *>(m_value.get());
        if (result == nullptr) {
          std::runtime_error("Incompatible Context type get cast.");
        }
        return result->get();
      }
      Platform get_platform() { return m_value->get_platform(); }
      template <typename T>
      T *allocate(size_t size)
      {
        return (T *)m_value->calloc(size * sizeof(T));
      }
      void *calloc(size_t size) { return m_value->calloc(size); }
      void free(void *p) { m_value->free(p); }
      void memcpy(void *dst, const void *src, size_t size)
      {
        m_value->memcpy(dst, src, size);
      }
      void memset(void *p, int val, size_t size)
      {
        m_value->memset(p, val, size);
      }
      Event get_event() { return m_value->get_event(); }
      void wait_on(Event *e) { m_value->wait_on(e); }

    private:
      class ContextInterface
      {
      public:
        virtual ~ContextInterface() {}
        virtual Platform get_platform() = 0;
        virtual void *calloc(size_t size) = 0;
        virtual void free(void *p) = 0;
        virtual void memcpy(void *dst, const void *src, size_t size) = 0;
        virtual void memset(void *p, int val, size_t size) = 0;
        virtual Event get_event() = 0;
        virtual void wait_on(Event *e) = 0;
      };

      template <typename T>
      class ContextModel : public ContextInterface
      {
      public:
        ContextModel(T const &modelVal) : m_modelVal(modelVal) {}
        Platform get_platform() override { return m_modelVal.get_platform(); }
        void *calloc(size_t size) override { return m_modelVal.calloc(size); }
        void free(void *p) override { m_modelVal.free(p); }
        void memcpy(void *dst, const void *src, size_t size) override
        {
          m_modelVal.memcpy(dst, src, size);
        }
        void memset(void *p, int val, size_t size) override
        {
          m_modelVal.memset(p, val, size);
        }
        Event get_event() { return m_modelVal.get_event_erased(); }
        void wait_on(Event *e) { m_modelVal.wait_on(e); }
        T get() { return m_modelVal; }

      private:
        T m_modelVal;
      };

      std::shared_ptr<ContextInterface> m_value;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif /* __CAMP_RESOURCE_HPP */
