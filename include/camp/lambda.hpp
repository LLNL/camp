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

#ifndef CAMP_LAMBDA_HPP
#define CAMP_LAMBDA_HPP

#include <type_traits>

#include "camp/defines.hpp"
#include "camp/list/at.hpp"
#include "camp/list/list.hpp"


namespace camp
{

template <template <typename...> class Expr>
struct lambda {
  template <typename... Ts>
  using expr = typename Expr<Ts...>::type;
};

template <typename Lambda, typename Seq>
struct apply_l;
template <typename Lambda, typename... Args>
struct apply_l<Lambda, list<Args...>> {
  using type = typename Lambda::template expr<Args...>::type;
};

template <typename Lambda, typename... Args>
struct invoke_l {
  using type = typename Lambda::template expr<Args...>::type;
};

template <idx_t n>
struct arg {
  template <typename... Ts>
  using expr = typename at<list<Ts...>, num<n - 1>>::type;
};

using _1 = arg<1>;
using _2 = arg<2>;
using _3 = arg<3>;
using _4 = arg<4>;
using _5 = arg<5>;
using _6 = arg<6>;
using _7 = arg<7>;
using _8 = arg<8>;
using _9 = arg<9>;

namespace detail
{
  template <typename T, typename... Args>
  struct get_bound_arg {
    using type = T;
  };
  template <idx_t i, typename... Args>
  struct get_bound_arg<arg<i>, Args...> {
    using type = typename arg<i>::template expr<Args...>;
  };
}  // namespace detail

template <template <typename...> class Expr, typename... ArgBindings>
struct bind {
  using bindings = list<ArgBindings...>;
  template <typename... Ts>
  using expr = typename Expr<
      typename detail::get_bound_arg<ArgBindings, Ts...>::type...>::type;
  using type = bind;
};

template <template <typename...> class Expr, typename... BoundArgs>
struct bind_front {
  template <typename... Ts>
  using expr = typename Expr<BoundArgs..., Ts...>::type;
  using type = bind_front;
};

CAMP_MAKE_L(bind_front);

}  // end namespace camp

#endif /* CAMP_LAMBDA_HPP */
