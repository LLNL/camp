//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "camp/camp.hpp"
#include "gtest/gtest.h"

using namespace camp;

TEST(CampTuple, AssignCompat)
{
  // Compatible, though different, tuples are assignable
  const tuple<long long, char> t(5, 'a');
  ASSERT_EQ(get<0>(t), 5);
  ASSERT_EQ(get<1>(t), 'a');

  tagged_tuple<list<int, char>, int, char> t2;
  t2 = t;
  ASSERT_EQ(get<0>(t2), 5);
  ASSERT_EQ(get<1>(t2), 'a');
}

TEST(CampTuple, Assign)
{
  tuple<int, char> t(5, 'a');
  ASSERT_EQ(get<0>(t), 5);
  ASSERT_EQ(get<1>(t), 'a');

  tuple<int, char> t2 = t;
  ASSERT_EQ(get<0>(t2), 5);
  ASSERT_EQ(get<1>(t2), 'a');
}

TEST(CampTuple, ForwardAsTuple)
{
  int a, b;
  [](tuple<int &, int &, int &&> t) {
    ASSERT_EQ(get<2>(t), 5);
    get<1>(t) = 3;
    get<2>(t) = 3;
    ASSERT_EQ(get<1>(t), 3);
    ASSERT_EQ(get<2>(t), 3);
  }(forward_as_tuple(a, b, int{5}));
}

TEST(CampTuple, GetByIndex)
{
  tuple<int, char> t(5, 'a');
  ASSERT_EQ(get<0>(t), 5);
  ASSERT_EQ(get<1>(t), 'a');
}

TEST(CampTuple, GetByType)
{
  tuple<int, char> t(5, 'a');
  ASSERT_EQ(get<int>(t), 5);
  ASSERT_EQ(get<char>(t), 'a');
}

TEST(CampTuple, CatPair)
{
  auto t1 = make_tuple(5, 'a');
  auto t2 = make_tuple(5.1f, "meh");
  auto t3 = tuple_cat_pair(t1,
                           make_idx_seq_t<2>{},
                           t2,
                           make_idx_seq_t<2>{});
  ASSERT_EQ(get<1>(t3), 'a');
  ASSERT_EQ(get<2>(t3), 5.1f);

  auto t4 = tuple_cat_pair(t1, t2);

  ASSERT_EQ(get<1>(t4), 'a');
  ASSERT_EQ(get<2>(t4), 5.1f);

  auto t5 =
      tuple_cat_pair(t1, idx_seq<1, 0>{}, t2, idx_seq<1, 0>{});
  ASSERT_EQ(get<0>(t5), 'a');
  ASSERT_EQ(get<3>(t5), 5.1f);
}

TEST(CampTuple, Default)
{
  tuple<int, float> t;
  auto t1 = tuple<int, float>{};
  auto t2 = tuple<int, float>();
  t = t1;
  t1 = t2;
}

struct NoDefCon {
  NoDefCon() = delete;
  NoDefCon(int i) : num{i} {(void)num;}
  NoDefCon(NoDefCon const &) = default;
  private:
  int num;
};

TEST(CampTuple, NoDefault)
{
  tuple<NoDefCon> t(NoDefCon(1));
}

struct s1;
struct s2;
struct s3;

TEST(CampTaggedTuple, GetByType)
{
  tagged_tuple<list<s1, s2>, int, char> t(5, 'a');
  ASSERT_EQ(get<s1>(t), 5);
  ASSERT_EQ(get<s2>(t), 'a');
  get<s1>(t) = 15;
  ASSERT_EQ(get<s1>(t), 15);
}

TEST(CampTaggedTuple, MakeTagged)
{
  auto t = make_tagged_tuple<list<s1, s2>>(5, 'a');
  ASSERT_EQ(get<s1>(t), 5);
  ASSERT_EQ(get<s2>(t), 'a');
  get<s1>(t) = 15;
  ASSERT_EQ(get<s1>(t), 15);
}
