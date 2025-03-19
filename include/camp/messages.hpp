#ifndef messages_HPP__
#define messages_HPP__

#include <algorithm>
#include <functional>
#include "camp/tuple.hpp"
#include "camp/resource.hpp"

namespace camp
{
  template <typename T>
  T atomic_fetch_inc(T* acc)
  {
    T ret = *acc;
    (*acc) += T(1) ;
    return ret;
  }

  template <typename... Args>
  using message = camp::tuple<std::decay_t<Args>...>;  

  template <typename... Args>
  class message_queue 
  {
  public:
    using value_type     = message<Args...> ;
    using size_type      = std::size_t;
    using pointer        = value_type*;
    using const_pointer  = const pointer;
    using iterator       = pointer;
    using const_iterator = const iterator;

    message_queue() : m_capacity{0}, m_size{0}, m_buf{nullptr} 
    {}
    message_queue(size_type capacity, pointer buf) : 
      m_capacity{capacity}, m_size{0}, m_buf{buf} 
    {}

    template <typename... Ts>
    bool try_emplace(Ts&&... args)
    {
      auto local_size = camp::atomic_fetch_inc(&m_size);
      if (m_buf != nullptr && local_size < m_capacity) {
        m_buf[local_size] = camp::make_tuple<Ts...>(std::forward<Ts>(args)...);
	return true;
      }

      return false;
    }

    pointer data() noexcept
    {
      return m_buf;
    }

    pointer data() const noexcept
    {
      return m_buf;
    }

    size_type capacity() const noexcept
    {
      return m_capacity;
    }

    size_type size() const noexcept
    {
      return std::min(m_capacity, m_size);
    }

    bool empty() const noexcept
    {
      return size() == 0;
    }

    iterator begin() noexcept { return data(); }
    iterator end() noexcept   { return data()+size(); }

    iterator begin() const noexcept { return data(); }
    iterator end() const noexcept   { return data()+size(); }

    iterator cbegin() const noexcept { return data(); }
    iterator cend() const noexcept   { return data()+size(); }

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

  template <typename R, typename... Args>
  class message_handler<R(Args...)>
  {
  public:
    using msg_queue     = message_queue<Args...>;
    using callback_type = std::function<R(Args...)>;

  public:
    template <typename Callable>
    message_handler(const std::size_t num_messages, Callable c) 
      : m_res{camp::resources::Host()}, 
        m_queue{num_messages, m_res.allocate<message<Args...>>(num_messages,
            camp::resources::MemoryAccess::Pinned)}, 
        m_callback{c}
    {}  

    template <typename Resource, typename Callable>
    message_handler(const std::size_t num_messages, Resource res, 
                    Callable c) 
      : m_res{res}, 
        m_queue{num_messages, m_res.allocate<message<Args...>>(num_messages,
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

    // Move ctor/operator
    message_handler(message_handler&&) = default;
    message_handler& operator=(message_handler&&) = default;

    template <typename... Ts>
    bool try_post_message(Ts&&... args)
    {
      return m_queue.try_emplace(std::forward<Ts>(args)...); 
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

#endif /* messages_HPP__ */
