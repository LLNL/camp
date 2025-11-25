//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __CAMP_DETAIL_TEST_HPP
#define __CAMP_DETAIL_TEST_HPP

#include "camp/type_traits/is_same.hpp"

namespace camp
{

///\cond
#ifndef CAMP_DOX
namespace test
{
  template <typename T1, typename T2>
  struct AssertSame {
    static_assert(is_same<T1, T2>::value,
                  "is_same assertion failed <see below for more information>");
    static bool constexpr value = is_same<T1, T2>::value;
  };

#define CAMP_UNQUOTE(...) __VA_ARGS__
#define CAMP_CHECK_SAME(X, Y)                                          \
  static_assert(                                                       \
      ::camp::test::AssertSame<CAMP_UNQUOTE X, CAMP_UNQUOTE Y>::value, \
      #X " same as " #Y)
#define CAMP_CHECK_TSAME(X, Y)                                          \
  static_assert(::camp::test::AssertSame<typename CAMP_UNQUOTE X::type, \
                                         CAMP_UNQUOTE Y>::value,        \
                #X " same as " #Y)

  template <typename Assertion, idx_t i>
  struct AssertValue {
    static_assert(Assertion::value == i,
                  "value assertion failed <see below for more information>");
    static bool const value = Assertion::value == i;
  };

#define CAMP_CHECK_IEQ(X, Y)                                            \
  static_assert(                                                        \
      ::camp::test::AssertValue<CAMP_UNQUOTE X, CAMP_UNQUOTE Y>::value, \
      #X "::value == " #Y)
}  // namespace test
#endif  // CAMP_DOX
///\endcond

}  // namespace camp


#endif /* __CAMP_DETAIL_TEST_HPP */
