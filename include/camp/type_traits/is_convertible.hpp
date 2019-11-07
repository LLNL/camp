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
  template <typename TO>
  constexpr void test_conversion(TO) noexcept;

}

/// type trait to validate T is convertible to U
template <typename T, typename U, typename V = void>
struct is_convertible : false_type {
};

/// type trait to validate T is convertible to U
template <typename FROM, typename TO>
struct is_convertible<FROM, TO, decltype(detail::test_conversion<TO>(val<FROM>()))> : true_type {
};

/// type trait to validate T is convertible to U
template <typename T, typename U>
using is_convertible_t = typename is_convertible<T,U>::type;

}  // namespace camp

#endif  // CAMP_IS_CONVERTIBLE_HPP
