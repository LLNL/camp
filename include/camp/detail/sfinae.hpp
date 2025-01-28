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

#ifndef CAMP_DETAIL_SFINAE_HPP
#define CAMP_DETAIL_SFINAE_HPP

#include "camp/helpers.hpp"
#include "camp/number/number.hpp"
#include "camp/value.hpp"

#include <type_traits>

namespace camp
{

/// \cond
namespace detail
{

  // caller pattern from metal library
  template <template <typename...> class expr, typename... vals>
  struct caller;

  template <
      template <typename...> class expr,
      typename... vals,
      typename std::enable_if<is_value<expr<vals...>>::value>::type* = nullptr>
  value<expr<vals...>> sfinae(caller<expr, vals...>*);

  value<> sfinae(...);

  template <template <typename...> class expr, typename... vals>
  struct caller : decltype(sfinae(declptr<caller<expr, vals...>>())) {
  };

  template <template <typename...> class Expr, typename... Vals>
  struct call_s : caller<Expr, Vals...> {
  };

  template <template <typename...> class Expr, typename... Vals>
  using call = Expr<Vals...>;
};  // namespace detail
/// \endcond

}  // end namespace camp

#endif /* CAMP_DETAIL_SFINAE_HPP */
