//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __CAMP_TYPE_TRAITS_IS_SAME_HPP
#define __CAMP_TYPE_TRAITS_IS_SAME_HPP

#include "camp/defines.hpp"
#include "camp/number/number.hpp"

namespace camp
{

template <typename T, typename U>
struct is_same_s : false_type {
};

template <typename T>
struct is_same_s<T, T> : true_type {
};

#if defined(CAMP_COMPILER_MSVC)
template <typename... Ts>
using is_same = typename is_same_s<Ts...>::type;
#else
template <typename T, typename U>
using is_same = typename is_same_s<T, U>::type;
#endif

template <typename T, typename U>
using is_same_t = is_same<T, U>;

}  // namespace camp

#endif /* __CAMP_TYPE_TRAITS_IS_SAME_HPP */
