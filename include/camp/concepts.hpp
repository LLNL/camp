/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_CONCEPTS_HPP
#define CAMP_CONCEPTS_HPP

#include <iterator>
#include <type_traits>

#include "camp/detector.hpp"
#include "camp/helpers.hpp"
#include "camp/list.hpp"
#include "camp/number.hpp"
#include "camp/type_traits/is_same.hpp"


#if defined(__cpp_concepts) && __cpp_concepts >= 202002L

#define CAMPC_CONCEPT_REQUIRES(name, args, ...) \
  template <typename... Ts>                     \
  concept name = requires args                  \
  {                                             \
    __VA_ARGS__;                                \
  }
#define CAMPC_CONVERTIBLE(TO, ...) \
  {                                \
    __VA_ARGS__;                   \
  } -> ::std::convertible_to<TO>
#define CAMPC_AND ;
#define CAMPC_CONCEPT concept

#else  // ^^^ concepts ^^^ / vvv no concepts vvv

#define CAMPC_CONCEPT_REQUIRES(name, args, ...)                         \
  auto CAMP_PP_CAT(name, _test_) args->decltype((__VA_ARGS__, void())); \
  template <typename... Ts>                                             \
  using CAMP_PP_CAT(name, _detector) =                                  \
      typename ::camp::concepts::detail::ret_type<                      \
          decltype(&CAMP_PP_CAT(name, _test_) < Ts... >)>::type;        \
  template <typename... Ts>                                             \
  constexpr bool name = is_detected_v<CAMP_PP_CAT(name, _detector), Ts...>

#define CAMPC_CONVERTIBLE(TO, ...) \
  ::camp::concepts::convertible_to_c<TO>(__VA_ARGS__)

#define CAMPC_AND ,
#define CAMPC_CONCEPT constexpr bool
#endif

namespace camp
{

namespace concepts
{
  namespace detail
  {
    template <typename T>
    struct ret_type;
    template <typename T, typename... Ts>
    struct ret_type<T (*)(Ts...)> {
      using type = T;
    };
  }  // namespace detail


  namespace metalib
  {
    using camp::is_same;

    /// negation metafunction of a value type
    template <typename T>
    struct negate_t : num<!T::value> {
    };

    /// all_of metafunction of a value type list -- all must be "true"
    template <bool... Bs>
    struct all_of : metalib::is_same<list<t, num<Bs>...>, list<num<Bs>..., t>> {
    };

    /// none_of metafunction of a value type list -- all must be "false"
    template <bool... Bs>
    struct none_of
        : metalib::is_same<idx_seq<false, Bs...>, idx_seq<Bs..., false>> {
    };

    /// any_of metafunction of a value type list -- at least one must be "true""
    template <bool... Bs>
    struct any_of : negate_t<none_of<Bs...>> {
    };

    /// all_of metafunction of a bool list -- all must be "true"
    template <typename... Bs>
    struct all_of_t : all_of<Bs::value...> {
    };

    /// none_of metafunction of a bool list -- all must be "false"
    template <typename... Bs>
    struct none_of_t : none_of<Bs::value...> {
    };

    /// any_of metafunction of a bool list -- at least one must be "true""
    template <typename... Bs>
    struct any_of_t : any_of<Bs::value...> {
    };

  }  // end namespace metalib

}  // end namespace concepts
}  // end namespace camp

namespace camp
{
namespace concepts
{

  namespace detail
  {
    // ------- Standard names, use standard library if available 

    /// concept to validate return type is convertible to given type
    template <typename from, typename to>
    CAMPC_CONCEPT convertible_to = std::is_convertible<from, to>::value;

    /// Constexpr helper for pre-c++20 convertible-to checks
    template <typename to, typename from>
    constexpr bool convertible_to_c(from &&)
    {
      return std::is_convertible<from, to>::value;
    }


    /// metafunction for use within decltype expression to validate type of
    /// expression
    template <typename from, typename to>
    CAMPC_CONCEPT same_as = ::camp::is_same<from, to>::value;

    template <typename T>
    CAMPC_CONCEPT_REQUIRES(swappable, (T && a, T &&b), swap(a, b));

    /// Basic equalitycomparable, NOTE: this is not complete c++20 compliant,
    /// but it is a reasonable proxy for now
    template <typename T>
    CAMPC_CONCEPT_REQUIRES(equality_comparable,
                           (T && a, T &&b),
                           convertible_to_c<bool>(a == b));

    template <class T, class U>
    CAMPC_CONCEPT_REQUIRES(
        equality_comparable_with,
        (const std::remove_reference_t<T> &a,
         const std::remove_reference_t<U> &b),
        equality_comparable<T> CAMPC_AND equality_comparable<U> CAMPC_AND
        convertible_to_c<bool>(a == b) CAMPC_AND
        convertible_to_c<bool>(a != b) CAMPC_AND
        convertible_to_c<bool>(b != a) CAMPC_AND
        convertible_to_c<bool>(b == a));

    // ---------- Non-standard names for convenience concepts
    template <typename T>
    CAMPC_CONCEPT_REQUIRES(less_than_comparable,
                           (T && a, T &&b),
                           convertible_to_c<bool>(a < b));

    template <typename T>
    CAMPC_CONCEPT_REQUIRES(greater_than_comparable,
                           (T && a, T &&b),
                           convertible_to_c<bool>(a > b));

    template <typename T>
    CAMPC_CONCEPT_REQUIRES(less_equal_comparable,
                           (T && a, T &&b),
                           convertible_to_c<bool>(a <= b));

    template <typename T>
    CAMPC_CONCEPT_REQUIRES(greater_equal_comparable,
                           (T && a, T &&b),
                           convertible_to_c<bool>(a >= b));

    template <class T, class U>
    CAMPC_CONCEPT_REQUIRES(
        comparable_with,
        (const std::remove_reference_t<T> &a,
         const std::remove_reference_t<U> &b),
        equality_comparable_with<T,U> CAMPC_AND
                        convertible_to_c<bool>(b < a)CAMPC_AND
                        convertible_to_c<bool>(a < b)CAMPC_AND
                        convertible_to_c<bool>(b <= a)CAMPC_AND
                        convertible_to_c<bool>(a <= b)CAMPC_AND
                        convertible_to_c<bool>(b > a)CAMPC_AND
                        convertible_to_c<bool>(a > b)CAMPC_AND
                        convertible_to_c<bool>(b >= a)CAMPC_AND
                        convertible_to_c<bool>(a >= b)CAMPC_AND
                        convertible_to_c<bool>(b == a)CAMPC_AND
                        convertible_to_c<bool>(a == b)CAMPC_AND
                        convertible_to_c<bool>(b != a)CAMPC_AND
                        convertible_to_c<bool>(a != b));

    template <typename T>
      CAMPC_CONCEPT comparable = comparable_with<T, T>;

    template <typename T>
      CAMPC_CONCEPT arithmetic = std::is_arithmetic<T>();

    template <typename T>
      CAMPC_CONCEPT FloatingPoint = std::is_floating_point<T>();

    template <typename T>
      CAMPC_CONCEPT Integral = std::is_integral<T>();

    template <typename T>
      CAMPC_CONCEPT signed_integral = Integral<T> && std::is_signed<T>();

    template <typename T>
      CAMPC_CONCEPT Unsigned = Integral<T> && std::is_unsigned<T>();

    template <typename T>
      struct Iterator
      : DefineConcept(is_not(Integral<T>()),  // hacky NVCC 8 workaround
          *(val<T>()),
          has_type<T &>(++val<T &>())) {
      };

    template <typename T>
      struct ForwardIterator
      : DefineConcept(Iterator<T>(), val<T &>()++, *val<T &>()++) {
      };

    template <typename T>
      struct BidirectionalIterator
      : DefineConcept(ForwardIterator<T>(),
          has_type<T &>(--val<T &>()),
          convertible_to<T const &>(val<T &>()--),
          *val<T &>()--) {
      };

    template <typename T>
      struct RandomAccessIterator
      : DefineConcept(BidirectionalIterator<T>(),
          Comparable<T>(),
          has_type<T &>(val<T &>() += val<diff_from<T>>()),
                        has_type<T>(val<T>() + val<diff_from<T>>()),
                        has_type<T>(val<diff_from<T>>() + val<T>()),
                        has_type<T &>(val<T &>() -= val<diff_from<T>>()),
                        has_type<T>(val<T>() - val<diff_from<T>>()),
                        val<T>()[val<diff_from<T>>()]) {
    };

    template <typename T>
    struct HasBeginEnd
        : DefineConcept(std::begin(val<T>()), std::end(val<T>())) {
    };

    template <typename T>
    struct Range
        : DefineConcept(HasBeginEnd<T>(), Iterator<iterator_from<T>>()) {
    };

    template <typename T>
    struct ForwardRange
        : DefineConcept(HasBeginEnd<T>(), ForwardIterator<iterator_from<T>>()) {
    };

    template <typename T>
    struct BidirectionalRange
        : DefineConcept(HasBeginEnd<T>(),
                        BidirectionalIterator<iterator_from<T>>()) {
    };

    template <typename T>
    struct RandomAccessRange
        : DefineConcept(HasBeginEnd<T>(),
                        RandomAccessIterator<iterator_from<T>>()) {
    };

  }  // end namespace concepts

  namespace type_traits
  {
    DefineTypeTraitFromConcept(is_iterator, camp::concepts::Iterator);
    DefineTypeTraitFromConcept(is_forward_iterator,
                               camp::concepts::ForwardIterator);
    DefineTypeTraitFromConcept(is_bidirectional_iterator,
                               camp::concepts::BidirectionalIterator);
    DefineTypeTraitFromConcept(is_random_access_iterator,
                               camp::concepts::RandomAccessIterator);

    DefineTypeTraitFromConcept(is_range, camp::concepts::Range);
    DefineTypeTraitFromConcept(is_forward_range, camp::concepts::ForwardRange);
    DefineTypeTraitFromConcept(is_bidirectional_range,
                               camp::concepts::BidirectionalRange);
    DefineTypeTraitFromConcept(is_random_access_range,
                               camp::concepts::RandomAccessRange);

    DefineTypeTraitFromConcept(is_comparable, camp::concepts::Comparable);
    DefineTypeTraitFromConcept(is_comparable_to, camp::concepts::ComparableTo);

    DefineTypeTraitFromConcept(is_arithmetic, camp::concepts::Arithmetic);
    DefineTypeTraitFromConcept(is_floating_point,
                               camp::concepts::FloatingPoint);
    DefineTypeTraitFromConcept(is_integral, camp::concepts::Integral);
    DefineTypeTraitFromConcept(is_signed, camp::concepts::Signed);
    DefineTypeTraitFromConcept(is_unsigned, camp::concepts::Unsigned);

    template <typename T>
    using IterableValue = decltype(*std::begin(camp::val<T>()));

    template <typename T>
    using IteratorValue = decltype(*camp::val<T>());

    namespace detail
    {

      /// \cond
      template <typename, template <typename...> class, typename...>
      struct IsSpecialized : camp::false_type {
      };

      template <template <typename...> class Template, typename... T>
      struct IsSpecialized<typename concepts::detail::voider<
                               decltype(camp::val<Template<T...>>())>::type,
                           Template,
                           T...> : camp::true_type {
      };

      template <template <class...> class,
                template <class...>
                class,
                bool,
                class...>
      struct SpecializationOf : camp::false_type {
      };

      template <template <class...> class Expected,
                template <class...>
                class Actual,
                class... Args>
      struct SpecializationOf<Expected, Actual, true, Args...>
          : camp::concepts::metalib::is_same<Expected<Args...>,
                                             Actual<Args...>> {
      };
      /// \endcond

    }  // end namespace detail


    template <template <class...> class Outer, class... Args>
    using IsSpecialized = detail::IsSpecialized<void, Outer, Args...>;

    template <template <class...> class, typename T>
    struct SpecializationOf : camp::false_type {
    };

    template <template <class...> class Expected,
              template <class...>
              class Actual,
              class... Args>
    struct SpecializationOf<Expected, Actual<Args...>>
        : detail::SpecializationOf<Expected,
                                   Actual,
                                   IsSpecialized<Expected, Args...>::value,
                                   Args...> {
    };

  }  // end namespace type_traits
}  // namespace camp

#endif /* CAMP_CONCEPTS_HPP */
