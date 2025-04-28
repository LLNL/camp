//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __CAMP_messages_hpp
#define __CAMP_messages_hpp

/*!
 * \file
 *
 * \brief   Support for storing messages on the device and perform callback on 
 * host.
 */

#include <algorithm>
#include <functional>
#include "camp/tuple.hpp"
#include "camp/resource.hpp"

namespace camp
{
  template <typename T>
  CAMP_HOST_DEVICE  
  T atomic_fetch_inc(T* acc)
  {
    // TODO need atomic operations for all systems here (OpenMP, SYCL)  
    // Due to the requirement of host device atomics, does this make sense 
    // to be moved to a differnet library?
    // The message queue currently is placing messages in a host device 
    // setting. This leads to more questions:
    // - Should the device and host be able to place messages at the same time?
    // - If not, should their be different implementations based on resource?
    //   (this would avoid need for HOST_DEVICE function)
    // - If so, would this need `*_system` level atomics on GPUs?  
    T old; 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) 
    old = atomicAdd(acc, T(1));
#else
    // Only works in the serial case
    old = *acc;
    (*acc) += T(1) ;
#endif
    return old;
  }

  ///
  /// Queue for storing messages. Fills buffer up to capacity.
  /// Once at capacity, messages are discarded.
  ///
  /// TODO: Currently, messages can be discarded. This is fine
  /// if the use case is storing the first error message(s). However,
  /// are there other use cases that need to read and write at 
  /// the same time?
  ///
  template <typename T>
  class message_queue 
  {
  public:
    using value_type     = T;
    using size_type      = unsigned long long;
    using pointer        = value_type*;
    using const_pointer  = const value_type*;
    using iterator       = pointer;
    using const_iterator = const_pointer;

    message_queue() : m_capacity{0}, m_size{0}, m_buf{nullptr} 
    {}
    message_queue(size_type capacity, pointer buf) : 
      m_capacity{capacity}, m_size{0}, m_buf{buf} 
    {}

    template <typename... Ts>
    CAMP_HOST_DEVICE
    bool try_emplace(Ts&&... args) {
      auto local_size = camp::atomic_fetch_inc(&m_size);
      if (m_buf != nullptr && local_size < m_capacity) {
        m_buf[local_size] = T(std::forward<Ts>(args)...);
	return true;
      }

      return false;
    }

    constexpr pointer data() noexcept {
      return m_buf;
    }

    constexpr const_pointer data() const noexcept {
      return m_buf;
    }

    constexpr size_type capacity() const noexcept {
      return m_capacity;
    }

    constexpr size_type size() const noexcept {
      return std::min(m_capacity, m_size);
    }

    constexpr bool empty() const noexcept {
      return size() == 0;
    }

    constexpr iterator begin() noexcept { 
      return data(); 
    }

    constexpr const_iterator begin() const noexcept { 
      return const_iterator(data()); 
    }

    constexpr const_iterator cbegin() const noexcept { 
      return const_iterator(data()); 
    }

    constexpr iterator end() noexcept {
      return data()+size(); 
    }

    constexpr const_iterator end() const noexcept { 
      return const_iterator(data()+size()); 
    }


    constexpr const_iterator cend() const noexcept   { 
      return const_iterator(data()+size()); 
    }

    void clear() noexcept
    {
      m_size = 0;
    }
  private:
    size_type m_capacity;
    size_type m_size;
    pointer m_buf;
  };

  template <typename Callable>
  class message_handler;

  ///
  /// Provides a way to handle messages from a GPU. This currently
  /// stores messages from the GPU and then calls a callback 
  /// function from the host.
  ///
  /// Note: 
  /// Currently, this forces a synchronize prior to calling 
  /// the callback function or testing if there are any messages.
  ///
  template <typename R, typename... Args>
  class message_handler<R(Args...)>
  {
  public:
    using message       = camp::tuple<std::decay_t<Args>...>;  
    using msg_queue     = message_queue<message>;
    using callback_type = std::function<R(Args...)>;

  public:
    template <typename Callable>
    message_handler(const std::size_t num_messages, Callable c) 
      : m_res{camp::resources::Host()}, 
        m_queue{num_messages, m_res.allocate<message>(num_messages,
            camp::resources::MemoryAccess::Pinned)}, 
        m_callback{c}
    {}  

    template <typename Resource, typename Callable>
    message_handler(const std::size_t num_messages, Resource res, 
                    Callable c) 
      : m_res{res}, 
        m_queue{num_messages, m_res.allocate<message>(num_messages,
            camp::resources::MemoryAccess::Pinned)}, 
        m_callback{c}
    {}  

    ~message_handler() 
    {
      m_res.wait();
      m_res.deallocate(m_queue.data(), camp::resources::MemoryAccess::Pinned); 
    }

    // Doesn't support copying 
    message_handler(const message_handler&) = delete;
    message_handler& operator=(const message_handler&) = delete;

    // TODO need proper move support 
    // Move ctor/operator
    message_handler(message_handler&&) = delete;
    message_handler& operator=(message_handler&&) = delete;

    template <typename... Ts>
    CAMP_HOST_DEVICE
    bool try_post_message(Ts&&... args)
    {
      return m_queue.try_emplace(camp::make_tuple(std::forward<Ts>(args)...)); 
    }

    void clear()
    {
      m_res.wait();   
      m_queue.clear();
    }

    bool test_any()
    {
      m_res.wait();   
      return !m_queue.empty(); 
    }

    void wait_all()
    {
      if (test_any()) {
        for (const auto& msg: m_queue) {
          camp::apply(m_callback, msg);     
        }
        clear();
      }
    }

  private:
    camp::resources::Resource m_res;
    msg_queue m_queue;
    callback_type m_callback;
  }; 
}

#endif /* __CAMP_messages_hpp */
