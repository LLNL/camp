//
// Created by Tom Scogland on 11/5/19.
//

#ifndef CAMP_IS_CONVERTIBLE_HPP
#define CAMP_IS_CONVERTIBLE_HPP

#include "../number/number.hpp"
#include "../helpers.hpp"

namespace camp
{

namespace detail
{
  template <typename T>
  constexpr void test_conversion(T &&u) noexcept;
}

/// type trait to validate T is convertible to U
template <typename T, typename U, typename V = void>
struct is_convertible : false_type {
};

/// type trait to validate T is convertible to U
template <typename T, typename U>
struct is_convertible<T, U, decltype(detail::test_conversion<U>(val<T>()))> : true_type {
};

/// type trait to validate T is convertible to U
template <typename T, typename U>
using is_convertible_t = typename is_convertible<T,U>::type;

}  // namespace camp

#endif  // CAMP_IS_CONVERTIBLE_HPP
