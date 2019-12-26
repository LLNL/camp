//
// Created by Tom Scogland on 11/5/19.
//

#include <camp/detail/test.hpp>
#include <camp/helpers.hpp>
#include <camp/list.hpp>
#include <camp/type_traits/detect.hpp>
#include <camp/concepts.hpp>

#include <list>
#include <vector>

using namespace camp;

struct A {
};
struct B : A {
};
struct C {
};
template <int I>
struct D : A {
};

template<template<typename...> class Op, class T, class _Res = decltype(is(Op<T>()))>
using invoke_is = _Res;

CAMP_DEF_CONCEPT_AND_TRAITS_T(Arith, is_arithmetic, std::is_arithmetic<T>::value);

// CAMP_DEF_CONCEPT_T(Arith, std::is_arithmetic<T>());
// CAMP_DEF_DETECTOR_T(Arith, invoke_is<std::is_arithmetic,T>);
static_assert(std::is_arithmetic<int>::value, "int is not arithmetic");
// static_assert(decltype(concepts::is(std::is_arithmetic<A>()))::value, "A is not arithmetic");
// static_assert(decltype(concepts::is(std::is_arithmetic<int>()))::value, "int is not arithmetic");
static_assert(CAMP_REQ(Arith, int), "int is arithmetic");
static_assert(is_arithmetic<int>::value, "int is arithmetic");
static_assert(!is_arithmetic<A>::value, "A is not arithmetic");

template<>
struct type_traits::is_index<B> : true_type {};
static_assert(type_traits::is_index<int>::value, "int is a valid indexing type");
static_assert(!type_traits::is_index<A>::value, "A is not a valid indexing type");
static_assert(type_traits::is_index<B>::value, "A is not a valid indexing type");
// CAMP_CHECK_VALUE_NOT(is_detected<Arith, A>);
// CAMP_CHECK_VALUE(is_detected<Arith, int>);
// CAMP_CHECK_VALUE(is_detected<Arith, float>);
// CAMP_CHECK_VALUE_NOT(is_detected<concepts::Arithmetic, A>);

static_assert(CAMP_REQ(concepts::comparable, int), "int can be compared with int");
static_assert(CAMP_REQ(concepts::equality_comparable, int), "int can be compared with int");
static_assert(CAMP_REQ(concepts::equality_comparable_with, int, long), "int and long can be compared");
static_assert(!CAMP_REQ(concepts::equality_comparable_with, A, long), "A and long can't be compared");

struct eq_only {
  bool operator==(eq_only const&o) {
    return true;
  }
  bool operator!=(eq_only const&o) {
    return false;
  }
};
static_assert(!CAMP_REQ(concepts::comparable, eq_only), "eq_only is not comparable");
static_assert(CAMP_REQ(concepts::equality_comparable, eq_only), "eq_only is equality comparable");

static_assert(!CAMP_REQ(concepts::iterator, eq_only), "eq_only is not an iterator");
static_assert(CAMP_REQ(concepts::iterator, int *), "int* is an iterator");
static_assert(CAMP_REQ(concepts::iterator, decltype(val<std::vector<int>>().begin())), "int* is an iterator");

static_assert(!CAMP_REQ(concepts::random_access_iterator, decltype(val<std::list<int>>().begin())), "list iterator is not random access");
static_assert(CAMP_REQ(concepts::random_access_iterator, int *), "int* is an iterator");
static_assert(CAMP_REQ(concepts::random_access_iterator, decltype(val<std::vector<int>>().begin())), "int* is an iterator");

static_assert(!CAMP_REQ(concepts::random_access_range, decltype(val<std::list<int>>())), "list iterator is not random access");
static_assert(CAMP_REQ(concepts::random_access_range, decltype(val<std::vector<int>>())), "int* is an iterator");

void bah (int, float);
CAMP_CHECK_VALUE_NOT(camp::concepts::invokable<decltype(bah), int, char*>);
CAMP_CHECK_VALUE(camp::concepts::invokable<decltype(bah), int, float>);
CAMP_CHECK_VALUE_NOT(camp::concepts::invokable_returns<decltype(bah), void, int, char*>);
CAMP_CHECK_VALUE(camp::concepts::invokable_returns<decltype(bah), void, int, float>);
CAMP_CHECK_VALUE_NOT(camp::concepts::invokable_returns<decltype(bah), int, int, float>);

struct TC1 {
  using my_member_type = C;

  static constexpr auto my_static_data_member = int(42);

  void my_method();

  int my_int_method(double, float);

  double my_method_with_overloads();

  B& my_method_with_overloads(my_member_type);

  void my_method_with_overloads(int**);

  my_member_type my_method_with_overloads(my_member_type, double);

  TC1& my_method_with_overloads(A const&);
};

// tests courtesy of Kokkos and David Hollman
CAMP_DEF_DETECTOR(_my_member_type_missing,
                  (class T),
                  typename T::my_member_type_missing);

CAMP_CHECK_VALUE_NOT(is_detected<_my_member_type_missing, TC1>);

//  template <class T>
//  using _my_member_type_archetype = typename T::my_member_type;
CAMP_DEF_DETECTOR_T(_my_member_type_archetype, typename T::my_member_type);

CAMP_CHECK_VALUE(is_detected<_my_member_type_archetype, TC1>);

CAMP_CHECK_VALUE(is_detected_exact<C, _my_member_type_archetype, TC1>);

CAMP_CHECK_VALUE(is_detected_convertible<C, _my_member_type_archetype, TC1>);


CAMP_DEF_REQUIREMENT_T(_my_method_archetype_1,T{}.my_method());

CAMP_DEF_DETECTOR_T(_my_method_archetype_2, decltype(declval<T>().my_method()));

CAMP_CHECK_VALUE(is_detected<_my_method_archetype_1, TC1>);

CAMP_CHECK_VALUE(is_detected<_my_method_archetype_2, TC1>);

CAMP_CHECK_VALUE(is_detected_exact<void, _my_method_archetype_1, TC1>);

CAMP_CHECK_VALUE(is_detected_exact<void, _my_method_archetype_2, TC1>);

template <class Tin>
struct OuterClass {
  // Things like this don't work with intel or cuda (probably a bug in the EDG
  // frontend)

  CAMP_DEF_DETECTOR_T(
      _inner_method_archetype,
      decltype(declval<T>().my_method_with_overloads(declval<Tin>())));

  CAMP_DEF_DETECTOR_TU(
      _inner_method_reversed_archetype_protected,
      decltype(declval<U>().my_method_with_overloads(declval<T>())));

  template <class T>
  struct _inner_method_reversed_archetype
      : detected_t<_inner_method_reversed_archetype_protected, T, Tin> {
  };

  // test the compiler's ability to handle indirection with this pattern
  // Should be the last overload when T = TC1 and Tin = C
  // (since the detected argument type should resolve to the second overload,
  //   which returns B&), and not detected otherwise
  CAMP_DEF_DETECTOR_T(_overload_nested_dependent_type_archetype,
                      decltype(declval<T>().my_method_with_overloads(
                          declval<detected_t<_inner_method_archetype, T>>())));
};

// Should be the third overload
CAMP_CHECK_VALUE(
    is_detected<OuterClass<int**>::template _inner_method_archetype, TC1>);
CAMP_CHECK_VALUE_NOT(
    is_detected<OuterClass<int>::template _inner_method_archetype, TC1>);

CAMP_CHECK_VALUE(
    std::is_same<
        void,
        detected_t<OuterClass<int**>::template _inner_method_archetype, TC1>>);

using OCL = OuterClass<list<std::integral_constant<int, 5>>>;

CAMP_CHECK_VALUE_NOT(is_detected_convertible<
                     TC1,
                     OCL::template _overload_nested_dependent_type_archetype,
                     TC1>);

CAMP_CHECK_VALUE_NOT(
    is_detected<OCL::template _overload_nested_dependent_type_archetype, TC1>);

// The hardest test: should be the last overload
CAMP_CHECK_VALUE(
    is_detected<
        OuterClass<C>::template _overload_nested_dependent_type_archetype,
        TC1>);

CAMP_CHECK_VALUE(is_convertible<TC1, TC1>);

CAMP_CHECK_VALUE(
    is_detected_convertible<
        TC1,
        OuterClass<C>::template _overload_nested_dependent_type_archetype,
        TC1>);

