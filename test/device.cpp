/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#include "camp/camp.hpp"
#include "gtest/gtest.h"


template <typename T>
class DeviceTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(DeviceTest);

TYPED_TEST_P(DeviceTest, Default)
{
  auto device = TypeParam::get_default();
  SUCCEED();
}

TYPED_TEST_P(DeviceTest, DefaultSync)
{
  auto device = TypeParam::get_default_sync();
  SUCCEED();
}

TYPED_TEST_P(DeviceTest, Get)
{
  auto device = TypeParam::get(0, nullptr);

  SUCCEED();
}

TYPED_TEST_P(DeviceTest, SyncAll)
{
  auto device = TypeParam::get_default();

  device.sync_all();

  SUCCEED();
}

TYPED_TEST_P(DeviceTest, Sync)
{
  auto device = TypeParam::get_default();

  device.sync();

  SUCCEED();
}

REGISTER_TYPED_TEST_CASE_P(
    DeviceTest, 
    Default,
    DefaultSync,
    Get,
    Sync,
    SyncAll
    );

INSTANTIATE_TYPED_TEST_CASE_P(Templated, DeviceTest, ::testing::Types<camp::devices::CudaDevice>);
