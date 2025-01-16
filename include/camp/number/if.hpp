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

#ifndef CAMP_NUMBER_IF_HPP
#define CAMP_NUMBER_IF_HPP

#include "camp/value.hpp"

#include <type_traits>

namespace camp
{

// TODO: document
template <bool Cond,
          typename Then = camp::true_type,
          typename Else = camp::false_type>
struct if_cs {
  using type = Then;
};

template <typename Then, typename Else>
struct if_cs<false, Then, Else> {
  using type = Else;
};

// TODO: document
template <bool Cond,
          typename Then = camp::true_type,
          typename Else = camp::false_type>
using if_c = typename if_cs<Cond, Then, Else>::type;

// TODO: document
template <typename Cond,
          typename Then = camp::true_type,
          typename Else = camp::false_type>
struct if_s : if_cs<Cond::value, Then, Else> {
};

template <typename Then, typename Else>
struct if_s<nil, Then, Else> : if_cs<false, Then, Else> {
};

// TODO: document
template <typename... Ts>
using if_ = typename if_s<Ts...>::type;

}  // end namespace camp

#endif /* CAMP_NUMBER_IF_HPP */
