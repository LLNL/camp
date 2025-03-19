//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "camp/messages.hpp"
#include "Test.hpp"

#include "gtest/gtest.h"

CAMP_TEST_BEGIN(message_handler, initialize) {
   int test = 0;
   camp::message_handler<void(int)> msg(1, [&](int val) {
     test = val;   
   });

   return !msg.test_any();
} CAMP_TEST_END(message_handler, initialize)

CAMP_TEST_BEGIN(message_handler, clear) {
   int test = 0;
   camp::message_handler<void(int)> msg(1, [&](int val) {
     test = val;   
   });
   msg.try_post_message(5);
   msg.clear();
   msg.wait_all();

   return test == 0;
} CAMP_TEST_END(message_handler, clear)

CAMP_TEST_BEGIN(message_handler, try_post_message) {
   int test = 0;
   camp::message_handler<void(int)> msg(1, [&](int val) {
     test = val;   
   });
   msg.try_post_message(5);
   msg.wait_all();

   return test == 5;
} CAMP_TEST_END(message_handler, try_post_message)
