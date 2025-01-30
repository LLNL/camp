//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef CAMP_NUMBER_NUMBER_HPP
#define CAMP_NUMBER_NUMBER_HPP

#include "camp/defines.hpp"

namespace camp
{

// TODO: document, consider making use/match std::integral_constant
template <class NumT, NumT v>
struct integral_constant {
  static constexpr NumT value = v;
  using value_type = NumT;
  using type = integral_constant;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

/**
 * @brief Short-form for a whole number
 *
 * @tparam N The integral value
 */
template <idx_t N>
using num = integral_constant<idx_t, N>;

using true_type = num<true>;
using false_type = num<false>;

using t = num<true>;

}  // end namespace camp

#endif /* CAMP_NUMBER_NUMBER_HPP */
