/*
Copyright (c) 2016-21, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_OPTIONAL_HPP
#define CAMP_OPTIONAL_HPP

#include <utility>
#include <type_traits>
#include <exception>
#include <initializer_list>

#include "camp/defines.hpp"


namespace camp
{

struct exception : std::exception
{
  exception() noexcept = default;

  exception(const exception& other) noexcept = default;

  exception& operator=(const exception& other) noexcept = default;

  virtual const char* what() const noexcept
  { return m_what; }

  virtual ~exception() = default;

protected:
  const char* m_what = "camp::exception";
};

struct bad_optional_access : camp::exception
{
  bad_optional_access() noexcept
    : m_what("camp::bad_optional_access")
  { }

  bad_optional_access( const bad_optional_access& other ) noexcept = default;

  bad_optional_access& operator=( const bad_optional_access& other ) noexcept = default;

  virtual const char* what() const noexcept
  { return m_what; }

  virtual ~bad_optional_access() = default;
};


struct nullopt_t {
  explicit constexpr nullopt_t(int) {}
};

inline constexpr nullopt_t nullopt(-1);


template < typename T >
struct optional
{
  using value_type = T;

  constexpr optional() noexcept = default;
  constexpr optional(camp::nullopt_t) noexcept : optional() { }

  constexpr optional(const optional& other) requires !std::is_copy_constructible<T>::value = delete;
  constexpr optional(const optional& other) requires std::is_trivially_copy_constructible<T>::value = default;

  constexpr optional(const optional& other) requires std::is_copy_constructible<T>::value &&
                                                     !std::is_trivially_copy_constructible<T>::value
  {
    if (other.has_value()) {
      emplace(*other);
    }
  }

  constexpr optional(optional&& other) noexcept
      requires std::is_trivially_move_constructible<T>::value = default;

  constexpr optional(optional&& other)
      noexcept(std::is_nothrow_move_constructible<T>::value)
      requires std::is_move_constructible<T>::value &&
               !std::is_trivially_move_constructible<T>::value
  {
    if (other.has_value()) {
      emplace(std::move(*other));
    }
  }




  CAMP_HOST_DEVICE
  optional& operator=(camp::nullopt_t) noexcept
  {
    reset();
    return *this;
  }

  CAMP_CONSTEXPR14 typename std::enable_if<!(std::is_copy_constructible<T>::value
                                          && std::is_copy_assignable<T>::value),
                            optional&>::type
  operator=(const optional& other) = delete;

  CAMP_CONSTEXPR14 typename std::enable_if<(std::is_trivially_copy_constructible<T>::value
                                         && std::is_trivially_copy_assignable<T>::value
                                         && std::is_trivially_destructible<T>::value),
                            optional&>::type
  operator=(const optional& other) = default;

  CAMP_HOST_DEVICE
  CAMP_CONSTEXPR14 typename std::enable_if<(std::is_copy_constructible<T>::value
                                         && std::is_copy_assignable<T>::value)
                                       && !(std::is_trivially_copy_constructible<T>::value
                                         && std::is_trivially_copy_assignable<T>::value
                                         && std::is_trivially_destructible<T>::value),
                            optional&>::type
  operator=(const optional& other)
  {
    if (m_has_value && other.has_value()) {
      **this = *other;
    } else if (m_has_value) {
      reset();
    } else if (other.has_value()) {
      emplace(*other);
    }
    return *this;
  }

  CAMP_CONSTEXPR14 typename std::enable_if<(std::is_move_constructible<T>::value
                                         && std::is_move_assignable<T>::value)
                                        && (std::is_trivially_move_constructible<T>::value
                                         && std::is_trivially_move_assignable<T>::value
                                         && std::is_trivially_destructible<T>::value),
                            optional&>::type
  operator=(optional&& other) = default;

  CAMP_HOST_DEVICE
  CAMP_CONSTEXPR14 typename std::enable_if<(std::is_move_constructible<T>::value
                                         && std::is_move_assignable<T>::value)
                                       && !(std::is_trivially_move_constructible<T>::value
                                         && std::is_trivially_move_assignable<T>::value
                                         && std::is_trivially_destructible<T>::value),
                            optional&>::type
  operator=(optional&& other)
      noexcept(std::is_nothrow_move_assignable<T>::value
            && std::is_nothrow_move_constructible<T>::value)
  {
    if (m_has_value && other.has_value()) {
      **this = std::move(*other);
    } else if (m_has_value) {
      reset();
    } else if (other.has_value()) {
      emplace(std::move(*other));
    }
    return *this;
  }

  CAMP_HOST_DEVICE
  template < typename U = T,
             typename = typename std::enable_if<
                 !std::is_same<
                   typename std::remove_cv<typename std::remove_reference<U>::type>::type,
                   camp::optional<T>>::value &&
                 std::is_constructible<T, U>::value &&
                 std::is_assignable<T&, U>::value &&
                   (!std::is_scalar<T>::value ||
                    !std::is_same<typename std::decay<U>::type, T>::value)>::type >
  optional& operator=(U&& value)
  {
    if (m_has_value) {
      **this = std::move(std::forward<U>(value));
    } else {
      emplace(std::forward<U>(value));
    }
    return *this;
  }

  CAMP_HOST_DEVICE
  template < typename U,
             typename = typename std::enable_if<
                 !(std::is_constructible<T, camp::optional<U>&>::value &&
                   std::is_constructible<T, const camp::optional<U>&>::value &&
                   std::is_constructible<T, camp::optional<U>&&>::value &&
                   std::is_constructible<T, const camp::optional<U>&&>::value &&
                   std::is_convertible<camp::optional<U>&, T>::value &&
                   std::is_convertible<const camp::optional<U>&, T>::value &&
                   std::is_convertible<camp::optional<U>&&, T>::value &&
                   std::is_convertible<const camp::optional<U>&&, T>::value &&
                   std::is_assignable<T&, camp::optional<U>&>::value &&
                   std::is_assignable<T&, const camp::optional<U>&>::value &&
                   std::is_assignable<T&, camp::optional<U>&&>::value &&
                   std::is_assignable<T&, const camp::optional<U>&&>::value) &&
                 std::is_constructible<T, const U&>::value &&
                 std::is_assignable<T&, const U&>::value >::type >
  optional& operator=(const optional<U>& other)
  {
    if (m_has_value && other.has_value()) {
      **this = *other;
    } else if (m_has_value) {
      reset();
    } else if (other.has_value()) {
      emplace(*other);
    }
    return *this;
  }

  CAMP_HOST_DEVICE
  template < typename U
             typename = typename std::enable_if<
                 !(std::is_constructible<T, camp::optional<U>&>::value &&
                   std::is_constructible<T, const camp::optional<U>&>::value &&
                   std::is_constructible<T, camp::optional<U>&&>::value &&
                   std::is_constructible<T, const camp::optional<U>&&>::value &&
                   std::is_convertible<camp::optional<U>&, T>::value &&
                   std::is_convertible<const camp::optional<U>&, T>::value &&
                   std::is_convertible<camp::optional<U>&&, T>::value &&
                   std::is_convertible<const camp::optional<U>&&, T>::value &&
                   std::is_assignable<T&, camp::optional<U>&>::value &&
                   std::is_assignable<T&, const camp::optional<U>&>::value &&
                   std::is_assignable<T&, camp::optional<U>&&>::value &&
                   std::is_assignable<T&, const camp::optional<U>&&>::value) &&
                 std::is_constructible<T, U>::value &&
                 std::is_assignable<T&, U>::value >::type >
  optional& operator=(optional<U>&& other)
  {
    if (m_has_value && other.has_value()) {
      **this = std::move(*other);
    } else if (m_has_value) {
      reset();
    } else if (other.has_value()) {
      emplace(std::move(*other));
    }
    return *this;
  }

  // have to break this up with partial specialization
  ~optional() requires std::is_trivially_destructible<T>::value = default;

  CAMP_HOST_DEVICE
  ~optional() requires !std::is_trivially_destructible<T>::value
  {
    reset();
  }

  CAMP_HOST_DEVICE
  constexpr explicit operator bool() const noexcept
  { return m_has_value; }
  CAMP_HOST_DEVICE
  constexpr bool has_value() const noexcept
  { return m_has_value; }

  CAMP_HOST_DEVICE
  constexpr const T* operator->() const
  { return reinterpret_cast<const T*>(&m_storage); }
  CAMP_HOST_DEVICE CAMP_CONSTEXPR14
                 T* operator->()
  { return reinterpret_cast<      T*>(&m_storage); }

  CAMP_HOST_DEVICE
  constexpr const  T& operator*() const&
  { return reinterpret_cast<const T&>(m_storage); }
  CAMP_HOST_DEVICE
  CAMP_CONSTEXPR14 T& operator*() &
  { return reinterpret_cast<      T&>(m_storage); }

  CAMP_HOST_DEVICE
  constexpr const  T&& operator*() const&&
  { return reinterpret_cast<const T&&>(m_storage); }
  CAMP_HOST_DEVICE
  CAMP_CONSTEXPR14 T&& operator*() &&
  { return reinterpret_cast<      T&&>(m_storage); }

  constexpr const  T& value() const&
  {
    return m_has_value ? reinterpret_cast<const T&>(m_storage)
                       : throw camp::bad_optional_access();
  }
  CAMP_CONSTEXPR14 T& value() &
  {
    return m_has_value ? reinterpret_cast<      T&>(m_storage)
                       : throw camp::bad_optional_access();
  }

  constexpr const  T&& value() const&&
  {
    return m_has_value ? reinterpret_cast<const T&&>(m_storage)
                       : throw camp::bad_optional_access();
  }
  CAMP_CONSTEXPR14 T&& value() &&
  {
    return m_has_value ? reinterpret_cast<      T&&>(m_storage)
                       : throw camp::bad_optional_access();
  }

  CAMP_HOST_DEVICE
  template < typename U >
  constexpr        T value_or(U&& default_value) const&
  {
    return m_has_value ? reinterpret_cast<const T&>(m_storage)
                       : static_cast<T>(std::forward<U>(default_value));
  }
  CAMP_HOST_DEVICE
  template < typename U >
  CAMP_CONSTEXPR14 T value_or(U&& default_value) &&
  {
    return m_has_value ? reinterpret_cast<      T&&>(m_storage)
                       : static_cast<T>(std::forward<U>(default_value));
  }

  void swap(optional& other)
#if defined(CAMP_HAS_SWAPPABLE_TRAITS)
      noexcept(std::is_nothrow_move_constructible<T>::value &&
               std::is_nothrow_swappable<T>::value)
#endif
  {
    if (m_has_value && other.has_value()) {
      using std::swap;
      swap(**this, *other);
    } else if (m_has_value) {
      other.emplace(std::move(**this));
      reset();
    } else if (other.has_value()) {
      emplace(std::move(*other));
      other.reset();
    }
  }

  CAMP_HOST_DEVICE
  void reset() noexcept
  {
    if (m_has_value) {
      (**this).T::~T();
      m_has_value = false;
    }
  }

  CAMP_HOST_DEVICE
  template < typename... Args >
  T& emplace(Args&&... args)
  {
    reset();
    new(static_cast<void*>(&m_storage)) T(std::forward<Args>(args)...);
    m_has_value = true;
    return **this;
  }
  template < typename U, typename... Args, typename = std::enable_if<
      std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value>::type >
  T& emplace(std::initializer_list<U> ilist, Args&&... args)
  {
    reset();
    new(static_cast<void*>(&m_storage)) T(ilist, std::forward<Args>(args)...);
    m_has_value = true;
    return **this;
  }

private:
  typename std::aligned_storage<sizeof(T), alignof(T)>::type m_storage;
  bool m_has_value = false;
};

}  // end namespace camp

#endif /* CAMP_OPTIONAL_HPP */
