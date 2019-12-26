/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_LIST_LIST_HPP
#define CAMP_LIST_LIST_HPP

#include "../number.hpp"
#include "../size.hpp"
#include "../type_traits/is_same.hpp"

namespace camp
{
// TODO: document

template <typename... Ts>
struct list {
  using type = list;
};

namespace detail
{
  template <typename T>
  struct _as_list;
  template <template <typename...> class T, typename... Args>
  struct _as_list<T<Args...>> {
    using type = list<Args...>;
  };
  template <typename T, T... Args>
  struct _as_list<int_seq<T, Args...>> {
    using type = list<integral_constant<T, Args>...>;
  };
}  // namespace detail

template <typename T>
struct as_list_s : detail::_as_list<T>::type {
};

template <typename T>
using as_list = typename as_list_s<T>::type;

template <typename... Args>
struct size<list<Args...>> {
  constexpr static idx_t value{sizeof...(Args)};
  using type = num<sizeof...(Args)>;
};

/// all_of metafunction of a value type list -- all must be "true"
#if defined(CAMP_HAS_FOLD_EXPRESSIONS)
template <bool... Bs>
struct all_of : num<(... && Bs)> {
};
#else
template <bool... Bs>
struct all_of : is_same<int_seq<bool, true, Bs...>, int_seq<bool, Bs..., true>> {
};
#endif

/// none_of metafunction of a value type list -- all must be "false"
#if defined(CAMP_HAS_FOLD_EXPRESSIONS)
template <bool... Bs>
struct none_of : num<!(... || Bs)> {
};
#else
template <bool... Bs>
struct none_of : is_same<int_seq<bool, false, Bs...>, int_seq<bool, Bs..., false>> {
};
#endif

/// any_of metafunction of a value type list -- at least one must be "true""
template <bool... Bs>
struct any_of : negate_t<none_of<Bs...>> {
};

/// all_of metafunction of a bool list -- all must be "true"
template <typename... Bs>
struct all_of_t : all_of<Bs::value...> {
};

/// none_of metafunction of a bool list -- all must be "false"
template <typename... Bs>
struct none_of_t : none_of<Bs::value...> {
};

/// any_of metafunction of a bool list -- at least one must be "true""
template <typename... Bs>
struct any_of_t : any_of<Bs::value...> {
};

}  // namespace camp

#endif /* CAMP_LIST_LIST_HPP */
