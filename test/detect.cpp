//
// Created by Tom Scogland on 11/5/19.
//

#include <camp/detail/test.hpp>
#include <camp/helpers.hpp>
#include <camp/list.hpp>
#include <camp/type_traits/detect.hpp>

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

// TODO test these in other scopes that could trip up compilers, like function
// scope or dependent class template scope

CAMP_DEF_DETECTOR(_my_member_type_missing,
                  (class T),
                  typename T::my_member_type_missing);

CAMP_CHECK_VALUE_NOT(is_detected<_my_member_type_missing, TC1>);

// tests courtesy of Kokkos and David Hollman
//  template <class T>
//  using _my_member_type_archetype = typename T::my_member_type;
CAMP_DEF_DETECTOR_T(_my_member_type_archetype, typename T::my_member_type);

CAMP_CHECK_VALUE(is_detected<_my_member_type_archetype, TC1>);

CAMP_CHECK_VALUE(is_detected_exact<C, _my_member_type_archetype, TC1>);

CAMP_CHECK_VALUE(is_detected_convertible<C, _my_member_type_archetype, TC1>);


CAMP_DEF_DETECTOR_T(_my_method_archetype_1, decltype(T{}.my_method()));

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
