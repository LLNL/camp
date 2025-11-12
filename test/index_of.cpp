//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((index_of<int, list<>>), (nil));
CAMP_CHECK_TSAME((index_of<int, list<float, double, int>>), (num<2>));
CAMP_CHECK_TSAME((index_of<int, list<float, double, int, int, int, int>>),
                 (num<2>));
