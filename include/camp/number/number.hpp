/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_NUMBER_NUMBER_HPP
#define CAMP_NUMBER_NUMBER_HPP

#include <type_traits>
#include "camp/defines.hpp"

namespace camp
{

/// re-export for backwards compatibility
using std::integral_constant;

/**
 * @brief Short-form for a whole number
 *
 * @tparam N The integral value
 */
template <idx_t N>
using num = integral_constant<idx_t, N>;

using true_type = std::true_type;
using false_type = std::false_type;

using t = num<true>;

}  // end namespace camp

#endif /* CAMP_NUMBER_NUMBER_HPP */
