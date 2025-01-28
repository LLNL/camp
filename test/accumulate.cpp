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

#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((accumulate<append, list<>, list<int, float, double>>),
                 (list<int, float, double>));
CAMP_CHECK_TSAME((cartesian_product<list<int>, list<float>>),
                 (list<list<int, float>>));
struct a;
struct b;
struct c;
struct d;
struct e;
struct f;
struct g;
CAMP_CHECK_TSAME((cartesian_product<list<a, b>, list<c, d, e>>),
                 (list<list<a, c>,
                       list<a, d>,
                       list<a, e>,
                       list<b, c>,
                       list<b, d>,
                       list<b, e>>));
CAMP_CHECK_TSAME((cartesian_product<list<a, b>, list<c, d, e>, list<f, g>>),
                 (camp::list<camp::list<a, c, f>,
                             camp::list<a, c, g>,
                             camp::list<a, d, f>,
                             camp::list<a, d, g>,
                             camp::list<a, e, f>,
                             camp::list<a, e, g>,
                             camp::list<b, c, f>,
                             camp::list<b, c, g>,
                             camp::list<b, d, f>,
                             camp::list<b, d, g>,
                             camp::list<b, e, f>,
                             camp::list<b, e, g>>));
