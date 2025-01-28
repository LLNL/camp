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

#ifndef CAMP_LIST_FIND_IF_HPP
#define CAMP_LIST_FIND_IF_HPP

#include <cstddef>
#include <type_traits>

#include "camp/lambda.hpp"
#include "camp/list/list.hpp"
#include "camp/number.hpp"
#include "camp/value.hpp"

namespace camp
{

/// \cond
namespace detail
{
  template <template <typename...> class Cond, typename... Elements>
  struct _find_if;
  template <template <typename...> class Cond, typename First, typename... Rest>
  struct _find_if<Cond, First, Rest...> {
    using type = if_<typename Cond<First>::type,
                     First,
                     typename _find_if<Cond, Rest...>::type>;
  };
  template <template <typename...> class Cond>
  struct _find_if<Cond> {
    using type = nil;
  };
}  // namespace detail
/// \endcond

template <template <typename...> class Cond, typename Seq>
struct find_if;

// TODO: document
template <template <typename...> class Cond, typename... Elements>
struct find_if<Cond, list<Elements...>> {
  using type = typename detail::_find_if<Cond, Elements...>::type;
};

CAMP_MAKE_L(find_if);

}  // end namespace camp

#endif /* CAMP_LIST_FIND_IF_HPP */
