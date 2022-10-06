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

#include <type_traits>

#include "camp/camp.hpp"
#include "gtest/gtest.h"

static_assert(std::is_same<camp::tuple<int &, int const &, int>,
                           decltype(camp::tuple_cat_pair(
                               camp::val<camp::tuple<int &>>(),
                               camp::val<camp::tuple<int const &, int>>()))>::value,
              "tuple_cat pair nuking refs");

static_assert(std::is_same<camp::tuple<int &&, int const &&, int>,
                           decltype(camp::tuple_cat_pair(
                               camp::val<camp::tuple<int &&>>(),
                               camp::val<camp::tuple<int const &&, int>>()))>::value,
              "tuple_cat pair nuking rvalue refs");

static_assert(std::is_same<camp::tuple<int &, int const &, int &&, int const &&>,
                           decltype(camp::forward_as_tuple(
                               camp::val<int &>(),
                               camp::val<int const &>(),
                               camp::val<int &&>(),
                               camp::val<int const &&>()))>::value,
              "forward_as_tuple nuking refs");

#define CAMP_TEST_TUPLE_GET_REFS(GET_REF, TYPE_REF, TUPLE_REF) \
static_assert(std::is_same<int GET_REF, \
                           decltype(camp::get<0>( \
                               camp::val<camp::tuple<int TYPE_REF> TUPLE_REF>()))>::value, \
              "get nuking refs");

#define CAMP_TEST_TUPLE_GET_T_REFS(GET_REF, TYPE_REF, TUPLE_REF) \
static_assert(std::is_same<int GET_REF, \
                           decltype(camp::get<int TYPE_REF>( \
                               camp::val<camp::tuple<int TYPE_REF> TUPLE_REF>()))>::value, \
              "get nuking refs");

#define CAMP_TEST_TAGGED_TUPLE_GET_REFS(GET_REF, TYPE_REF, TUPLE_REF) \
static_assert(std::is_same<int GET_REF, \
                           decltype(camp::get<0>( \
                               camp::val<camp::tagged_tuple<camp::list<float>, int TYPE_REF> TUPLE_REF>()))>::value, \
              "get nuking refs");

#define CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(GET_REF, TYPE_REF, TUPLE_REF) \
static_assert(std::is_same<int GET_REF, \
                           decltype(camp::get<float>( \
                               camp::val<camp::tagged_tuple<camp::list<float>, int TYPE_REF> TUPLE_REF>()))>::value, \
              "get nuking refs");

CAMP_TEST_TUPLE_GET_REFS(     & ,        ,      & )
CAMP_TEST_TUPLE_GET_REFS(     &&,        ,      &&)
CAMP_TEST_TUPLE_GET_REFS(const& ,        , const& )
CAMP_TEST_TUPLE_GET_REFS(const&&,        , const&&)

CAMP_TEST_TUPLE_GET_REFS(     & ,      & ,      & )
CAMP_TEST_TUPLE_GET_REFS(     & ,      & ,      &&)
CAMP_TEST_TUPLE_GET_REFS(     & ,      & , const& )
CAMP_TEST_TUPLE_GET_REFS(     & ,      & , const&&)

CAMP_TEST_TUPLE_GET_REFS(const& , const& ,      & )
CAMP_TEST_TUPLE_GET_REFS(const& , const& ,      &&)
CAMP_TEST_TUPLE_GET_REFS(const& , const& , const& )
CAMP_TEST_TUPLE_GET_REFS(const& , const& , const&&)

CAMP_TEST_TUPLE_GET_REFS(     & ,      &&,      & )
CAMP_TEST_TUPLE_GET_REFS(     &&,      &&,      &&)
CAMP_TEST_TUPLE_GET_REFS(     & ,      &&, const& )
CAMP_TEST_TUPLE_GET_REFS(     &&,      &&, const&&)

CAMP_TEST_TUPLE_GET_REFS(const& , const&&,      & )
CAMP_TEST_TUPLE_GET_REFS(const&&, const&&,      &&)
CAMP_TEST_TUPLE_GET_REFS(const& , const&&, const& )
CAMP_TEST_TUPLE_GET_REFS(const&&, const&&, const&&)


CAMP_TEST_TUPLE_GET_T_REFS(     & ,        ,      & )
CAMP_TEST_TUPLE_GET_T_REFS(     &&,        ,      &&)
CAMP_TEST_TUPLE_GET_T_REFS(const& ,        , const& )
CAMP_TEST_TUPLE_GET_T_REFS(const&&,        , const&&)

CAMP_TEST_TUPLE_GET_T_REFS(     & ,      & ,      & )
CAMP_TEST_TUPLE_GET_T_REFS(     & ,      & ,      &&)
CAMP_TEST_TUPLE_GET_T_REFS(     & ,      & , const& )
CAMP_TEST_TUPLE_GET_T_REFS(     & ,      & , const&&)

CAMP_TEST_TUPLE_GET_T_REFS(const& , const& ,      & )
CAMP_TEST_TUPLE_GET_T_REFS(const& , const& ,      &&)
CAMP_TEST_TUPLE_GET_T_REFS(const& , const& , const& )
CAMP_TEST_TUPLE_GET_T_REFS(const& , const& , const&&)

CAMP_TEST_TUPLE_GET_T_REFS(     & ,      &&,      & )
CAMP_TEST_TUPLE_GET_T_REFS(     &&,      &&,      &&)
CAMP_TEST_TUPLE_GET_T_REFS(     & ,      &&, const& )
CAMP_TEST_TUPLE_GET_T_REFS(     &&,      &&, const&&)

CAMP_TEST_TUPLE_GET_T_REFS(const& , const&&,      & )
CAMP_TEST_TUPLE_GET_T_REFS(const&&, const&&,      &&)
CAMP_TEST_TUPLE_GET_T_REFS(const& , const&&, const& )
CAMP_TEST_TUPLE_GET_T_REFS(const&&, const&&, const&&)


CAMP_TEST_TAGGED_TUPLE_GET_REFS(     & ,        ,      & )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(     &&,        ,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_REFS(const& ,        , const& )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(const&&,        , const&&)

CAMP_TEST_TAGGED_TUPLE_GET_REFS(     & ,      & ,      & )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(     & ,      & ,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_REFS(     & ,      & , const& )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(     & ,      & , const&&)

CAMP_TEST_TAGGED_TUPLE_GET_REFS(const& , const& ,      & )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(const& , const& ,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_REFS(const& , const& , const& )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(const& , const& , const&&)

CAMP_TEST_TAGGED_TUPLE_GET_REFS(     & ,      &&,      & )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(     &&,      &&,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_REFS(     & ,      &&, const& )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(     &&,      &&, const&&)

CAMP_TEST_TAGGED_TUPLE_GET_REFS(const& , const&&,      & )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(const&&, const&&,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_REFS(const& , const&&, const& )
CAMP_TEST_TAGGED_TUPLE_GET_REFS(const&&, const&&, const&&)


CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     & ,        ,      & )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     &&,        ,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const& ,        , const& )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const&&,        , const&&)

CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     & ,      & ,      & )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     & ,      & ,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     & ,      & , const& )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     & ,      & , const&&)

CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const& , const& ,      & )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const& , const& ,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const& , const& , const& )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const& , const& , const&&)

CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     & ,      &&,      & )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     &&,      &&,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     & ,      &&, const& )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(     &&,      &&, const&&)

CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const& , const&&,      & )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const&&, const&&,      &&)
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const& , const&&, const& )
CAMP_TEST_TAGGED_TUPLE_GET_T_REFS(const&&, const&&, const&&)


// Size tests, ensure that EBO is being applied
struct A {
};
struct B {
};

// These are off by default, because failing this is not a fatal condition
#ifdef TEST_EBO
static_assert(sizeof(camp::tuple<A, B>) == 1, "EBO working as intended with empty types");

static_assert(sizeof(camp::tuple<A, B, ptrdiff_t>) == sizeof(ptrdiff_t),
              "EBO working as intended with one sized type at the end");

static_assert(sizeof(camp::tuple<A, ptrdiff_t, B>) == sizeof(ptrdiff_t),
              "EBO working as intended with one sized type in the middle");
#endif // TEST_EBO

// is_empty on all empty members currently is not true, same as libc++, though
// libstdc++ supports it.  This could be fixed by refactoring base member into a
// base class, but makes certain things messy and may have to be public, not
// clear it's worth it, either way the size of one tuple<A,B> should be 1 as it
// is in both libc++ and libstdc++
// static_assert(std::is_empty<camp::tuple<A, B>>::value, "it's empty right?");

// Ensure trivial copyability for trivially copyable contents
#if CAMP_HAS_IS_TRIVIALLY_COPY_CONSTRUCTIBLE
static_assert(
    std::is_trivially_copy_constructible<camp::tuple<int, float>>::value,
    "can by trivially copy constructed");
#endif

// Execution tests

TEST(CampTuple, AssignCompat)
{
  // Compatible, though different, tuples are assignable
  const camp::tuple<int, char> t(5, 'a');
  ASSERT_EQ(camp::get<0>(t), 5);
  ASSERT_EQ(camp::get<1>(t), 'a');

  camp::tagged_tuple<camp::list<int, char>, long long, char> t2;
  t2 = t;
  ASSERT_EQ(camp::get<0>(t2), 5);
  ASSERT_EQ(camp::get<1>(t2), 'a');
  camp::tagged_tuple<camp::list<int, char>, short, char> t3(5, 13);
  t2 = t3;
}

TEST(CampTuple, Assign)
{
  camp::tuple<int, char> t(5, 'a');
  ASSERT_EQ(camp::get<0>(t), 5);
  ASSERT_EQ(camp::get<1>(t), 'a');

  camp::tuple<int, char> t2 = t;
  ASSERT_EQ(camp::get<0>(t2), 5);
  ASSERT_EQ(camp::get<1>(t2), 'a');
}

TEST(CampTuple, ForwardAsTuple)
{
  int a, b;
  [](camp::tuple<int &, int &, int &&> t) {
    ASSERT_EQ(camp::get<2>(t), 5);
    camp::get<1>(t) = 3;
    camp::get<2>(t) = 3;
    ASSERT_EQ(camp::get<1>(t), 3);
    ASSERT_EQ(camp::get<2>(t), 3);
  }(camp::forward_as_tuple(a, b, int{5}));
}

TEST(CampTuple, GetByIndex)
{
  camp::tuple<int, char> t(5, 'a');
  ASSERT_EQ(camp::get<0>(t), 5);
  ASSERT_EQ(camp::get<1>(t), 'a');
}

TEST(CampTuple, GetByType)
{
  camp::tuple<int, char> t(5, 'a');
  ASSERT_EQ(camp::get<int>(t), 5);
  ASSERT_EQ(camp::get<char>(t), 'a');
}

TEST(CampTuple, CatPair)
{
  auto t1 = camp::make_tuple(5, 'a');
  auto t2 = camp::make_tuple(5.1f, std::string("meh"));
  auto t3 = tuple_cat_pair(t1,
                           camp::make_idx_seq_t<2>{},
                           t2,
                           camp::make_idx_seq_t<2>{});
  ASSERT_EQ(camp::get<1>(t3), 'a');
  ASSERT_EQ(camp::get<2>(t3), 5.1f);

  auto t4 = tuple_cat_pair(t1, t2);

  ASSERT_EQ(camp::get<1>(t4), 'a');
  ASSERT_EQ(camp::get<2>(t4), 5.1f);

  auto t5 =
      tuple_cat_pair(t1, camp::idx_seq<1, 0>{}, t2, camp::idx_seq<1, 0>{});
  ASSERT_EQ(camp::get<0>(t5), 'a');
  ASSERT_EQ(camp::get<3>(t5), 5.1f);
}

struct NoDefCon {
  NoDefCon() = delete;
  NoDefCon(int i) : num{i} { (void)num; }
  NoDefCon(NoDefCon const &) = default;

private:
  int num;
};

TEST(CampTuple, NoDefault) { camp::tuple<NoDefCon> t(NoDefCon(1)); }

int testFunctionWithoutArgs()
{
  return 42;
}

struct TestFunctor {
  double operator()(double value)
  {
    return value;
  }
};

auto testLambda = [] (int value1, double value2)
{
  return value1 + value2;
};

TEST(CampTuple, Apply)
{
  auto t1 = camp::make_tuple();
  ASSERT_EQ(camp::apply(testFunctionWithoutArgs, t1), 42);

  auto t2 = camp::make_tuple(8.1);
  ASSERT_NEAR(camp::apply(TestFunctor{}, t2), 8.1, 1e-15);

  auto t3 = camp::make_tuple(2, 5.5);
  ASSERT_NEAR(camp::apply(testLambda, t3), 7.5, 1e-15);
}

#if defined(__cplusplus) && __cplusplus >= 201703L
TEST(CampTuple, ClassTemplateArgumentDeduction)
{
  camp::tuple t{3, 9.9};
  ASSERT_EQ(camp::get<0>(t), 3);
  ASSERT_NEAR(camp::get<1>(t), 9.9, 1e-15);
}

TEST(CampTuple, StructuredBindings)
{
  auto t = camp::make_tuple(3, 9.9);
  auto [a, b] = t;
  ASSERT_EQ(a, 3);
  ASSERT_NEAR(b, 9.9, 1e-15);
}
#endif

struct s1;
struct s2;
struct s3;

TEST(CampTaggedTuple, GetByType)
{
  camp::tagged_tuple<camp::list<s1, s2>, int, char> t(5, 'a');
  ASSERT_EQ(camp::get<s1>(t), 5);
  ASSERT_EQ(camp::get<s2>(t), 'a');
  camp::get<s1>(t) = 15;
  ASSERT_EQ(camp::get<s1>(t), 15);
}

TEST(CampTaggedTuple, MakeTagged)
{
  auto t = camp::make_tagged_tuple<camp::list<s1, s2>>(5, 'a');
  ASSERT_EQ(camp::get<s1>(t), 5);
  ASSERT_EQ(camp::get<s2>(t), 'a');
  camp::get<s1>(t) = 15;
  ASSERT_EQ(camp::get<s1>(t), 15);
}

#if defined(__cplusplus) && __cplusplus >= 201703L
TEST(CampTaggedTuple, StructuredBindings)
{
  auto t = camp::make_tagged_tuple<camp::list<s1, s2>>('a', 5);
  auto [a, b] = t;
  ASSERT_EQ(a, 'a');
  ASSERT_EQ(b, 5);
}
#endif
