//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "camp/array.hpp"
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

CAMP_TEST_BEGIN(message_handler, initialize_with_resource) {
   int test = 0;
   camp::message_handler<void(int)> msg(1, camp::resources::Host(), [&](int val) {
     test = val;   
   });

   return !msg.test_any();
} CAMP_TEST_END(message_handler, initialize_with_resource)

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

   return msg.try_post_message(5);
} CAMP_TEST_END(message_handler, try_post_message)

CAMP_TEST_BEGIN(message_handler, try_post_message_overflow) {
   int test = 0;
   camp::message_handler<void(int)> msg(1, [&](int val) {
     test = val;   
   });
   msg.try_post_message(5);

   return !msg.try_post_message(6);
} CAMP_TEST_END(message_handler, try_post_message_overflow)

CAMP_TEST_BEGIN(message_handler, wait_all) {
   int test = 0;
   camp::message_handler<void(int)> msg(1, [&](int val) {
     test = val;   
   });
   msg.try_post_message(1);
   msg.wait_all();

   return test == 1;
} CAMP_TEST_END(message_handler, wait_all)

CAMP_TEST_BEGIN(message_handler, wait_all_array) {
   camp::array<int, 3> test = {0, 0, 0};
   camp::message_handler<void(camp::array<int, 3>)> msg(1, 
     [&](camp::array<int, 3> val) {
       test[0] = val[0];   
       test[1] = val[1];
       test[2] = val[2];
     }
   );
   camp::array<int, 3> a{1,2,3};
   msg.try_post_message(a);
   msg.wait_all();

   return test[0] == 1 && 
          test[1] == 2 && 
          test[2] == 3;
} CAMP_TEST_END(message_handler, wait_all_array)

CAMP_TEST_BEGIN(message_handler, wait_all_overflow) {
   int test = 0;
   camp::message_handler<void(int)> msg(1, [&](int val) {
     test = val;   
   });
   msg.try_post_message(1);
   msg.try_post_message(2);
   msg.wait_all();

   return test == 1;
} CAMP_TEST_END(message_handler, wait_all_overflow)
