//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_IEQ((size<list<int>>), (1));
CAMP_CHECK_IEQ((size<list<int, int>>), (2));
CAMP_CHECK_IEQ((size<list<int, int, int>>), (3));


CAMP_CHECK_IEQ((size<idx_seq<0>>), (1));
CAMP_CHECK_IEQ((size<idx_seq<0, 0>>), (2));
CAMP_CHECK_IEQ((size<idx_seq<0, 0, 0>>), (3));
