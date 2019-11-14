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

#include "helpers.hpp"
#include "list.hpp"
#include "number.hpp"

#include "type_traits/detect.hpp"
#include "type_traits/enable_if.hpp"
#include "type_traits/is_same.hpp"

namespace camp
{

namespace concepts
{

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

#define DefineConcept(...) decltype((__VA_ARGS__, true_type()))

#define DefineTypeTraitFromConcept(TTName, ConceptName)       \
  template <typename... Args>                                 \
  struct TTName : ::camp::is_detected<ConceptName, Args...> { \
  }
namespace camp
{
namespace concepts
{

  namespace detail
  {
    template <typename Ret, typename T>
    Ret returns(T const &) noexcept;

  }  // end namespace detail

  template <typename T>
  using negate = metalib::negate_t<T>;

  /// metafunction for use within decltype expression to validate return type is
  /// convertible to given type
  template <typename T, typename U>
  constexpr auto convertible_to(U &&u) noexcept
      -> decltype(detail::returns<camp::true_type>(static_cast<T>((U &&) u)));

  /// metafunction for use within decltype expression to validate type of
  /// expression
  template <typename T, typename U>
  constexpr auto has_type(U &&) noexcept -> metalib::is_same<T, U>;

  template <typename BoolLike>
  constexpr auto is(BoolLike) noexcept
      -> camp::enable_if_t<BoolLike::value, true_type>
  {
    return {};
  }

  template <typename BoolLike, camp::enable_if_t<!BoolLike::value, int> = 0>
  constexpr true_type is_not(BoolLike) noexcept
  {
    return {};
  }

  /// metaprogramming concept for SFINAE checking of aggregating concepts
  template <typename... Args>
  struct all_of : metalib::all_of_t<Args...> {
  };

  /// metaprogramming concept for SFINAE checking of aggregating concepts
  template <typename... Args>
  struct none_of : metalib::none_of_t<Args...> {
  };

  /// metaprogramming concept for SFINAE checking of aggregating concepts
  template <typename... Args>
  struct any_of : metalib::any_of_t<Args...> {
  };

  /// SFINAE concept checking
  template <template <class...> class Op, class... Args>
  struct requires_ : is_detected<Op, Args...> {
  };

  // TODO: add a proper ranges-style swap and update this
  CAMP_DEF_REQUIREMENT_T(Swappable, swap(val<T>(), val<T>()));

  CAMP_DEF_REQUIREMENT_TU(LessThan, val<T>() < val<U>());
  CAMP_DEF_REQUIREMENT_TU(GreaterThan, val<T>() > val<U>());
  CAMP_DEF_REQUIREMENT_TU(LessEqual, val<T>() <= val<U>());
  CAMP_DEF_REQUIREMENT_TU(GreaterEqual, val<T>() >= val<U>());

  CAMP_DEF_REQUIREMENT_TU(Equality, val<T>() == val<U>());
  CAMP_DEF_REQUIREMENT_TU(Inequality, val<T>() != val<U>());

  CAMP_DEF_CONCEPT_TU(__Weakly_equality_comparable_with,
                      (detect_convertible<bool, Equality, T, U>()
                       && detect_convertible<bool, Inequality, T, U>()
                       && detect_convertible<bool, Equality, U, T>()
                       && detect_convertible<bool, Inequality, U, T>()));

  CAMP_DEF_CONCEPT_AND_TRAITS_T(equality_comparable,
                                is_equality_comparable,
                                CAMP_REQ(__Weakly_equality_comparable_with,
                                         T,
                                         T));

  CAMP_DEF_CONCEPT_AND_TRAITS_TU(
      equality_comparable_with,
      is_equality_comparable_with,
      CAMP_REQ(equality_comparable, T) && CAMP_REQ(equality_comparable, U)
          && CAMP_REQ(__Weakly_equality_comparable_with, T, U));

  CAMP_DEF_CONCEPT_AND_TRAITS_TU(
      comparable_with,
      is_comparable_with,
      CAMP_REQ(equality_comparable_with, T, U)
          && detect_convertible<bool, LessThan, T, U>()
          && detect_convertible<bool, GreaterThan, T, U>()
          && detect_convertible<bool, LessEqual, T, U>()
          && detect_convertible<bool, GreaterEqual, T, U>()
          && detect_convertible<bool, LessThan, U, T>()
          && detect_convertible<bool, GreaterThan, U, T>()
          && detect_convertible<bool, LessEqual, U, T>()
          && detect_convertible<bool, GreaterEqual, U, T>());

  CAMP_DEF_CONCEPT_AND_TRAITS_T(comparable,
                                is_comparable,
                                CAMP_REQ(comparable_with, T, T));

  CAMP_DEF_CONCEPT_AND_TRAITS_T(arithmetic,
                                is_arithmetic,
                                std::is_arithmetic<T>::value);
  CAMP_DEF_CONCEPT_AND_TRAITS_T(floating_point,
                                is_floating_point,
                                std::is_floating_point<T>::value);
  CAMP_DEF_CONCEPT_AND_TRAITS_T(integral,
                                is_integral,
                                std::is_integral<T>::value);
  CAMP_DEF_CONCEPT_AND_TRAITS_T(signed, is_signed, std::is_signed<T>::value);
  CAMP_DEF_CONCEPT_AND_TRAITS_T(unsigned,
                                is_unsigned,
                                std::is_unsigned<T>::value);

  CAMP_DEF_REQUIREMENT_T(Dereference, *(val<T>()));
  CAMP_DEF_REQUIREMENT_T(IncrementPre, ++val<T>());
  CAMP_DEF_CONCEPT_AND_TRAITS_T(iterator,
                                is_iterator,
                                detect<Dereference, T>()
                                    && detect_exact<T &, IncrementPre, T &>());

  CAMP_DEF_REQUIREMENT_T(IncrementPost, val<T>()++);
  CAMP_DEF_REQUIREMENT_T(DereferenceIncrementPost, *val<T>()++);
  CAMP_DEF_CONCEPT_AND_TRAITS_T(forward_iterator,
                                is_forward_iterator,
                                CAMP_REQ(iterator, T)
                                    && detect<IncrementPost, T &>()
                                    && detect<DereferenceIncrementPost, T &>());

  CAMP_DEF_REQUIREMENT_T(DecrementPre, --val<T>());
  CAMP_DEF_REQUIREMENT_T(DecrementPost, val<T>()--);
  CAMP_DEF_REQUIREMENT_T(DereferenceDecrementPost, *val<T>()--);
  CAMP_DEF_CONCEPT_AND_TRAITS_T(
      bidirectional_iterator,
      is_bidirectional_iterator,
      CAMP_REQ(forward_iterator, T) && detect_exact<T &, DecrementPre, T &>()
          && detect_convertible<T const &, DecrementPost, T &>()
          && detect<DereferenceDecrementPost, T &>());

  CAMP_DEF_REQUIREMENT_T(MemberDifferenceType, T::difference_type);
  CAMP_DEF_REQUIREMENT_TU(PlusEq, val<T>() += val<U>());
  CAMP_DEF_REQUIREMENT_TU(Plus, val<T>() + val<U>());
  CAMP_DEF_REQUIREMENT_TU(SubEq, val<T>() -= val<U>());
  CAMP_DEF_REQUIREMENT_TU(Sub, val<T>() - val<U>());
  CAMP_DEF_REQUIREMENT_TU(Index, val<T>()[val<U>()]);
  template <class I, typename = void>
  struct incrementable_traits {
  };
  template <class T>
  struct incrementable_traits<T *, enable_if_t<std::is_object<T>::value>> {
    using difference_type = ptrdiff_t;
  };
  template <class T>
  struct incrementable_traits<const T> : incrementable_traits<T> {
  };
  template <class T>
  struct incrementable_traits<
      T,
      enable_if_t<detect<MemberDifferenceType, T>(), void>> {
    using difference_type = typename T::difference_type;
  };
  template <class T>
  struct incrementable_traits<
      T,
      enable_if_t<(!std::is_pointer<T>::value)
                      && (!detect<MemberDifferenceType, T>())
                      && detect_convertible<ptrdiff_t, Sub, T, T>(),
                  void>> {
    using difference_type = decltype(val<plain<T>>() - val<plain<T>>());
  };
  template <typename T>
  using difft_from = typename incrementable_traits<T>::difference_type;

  CAMP_DEF_REQUIREMENT_T(ItPlusEqDiff, val<T>() += val<difft_from<T>>());
  CAMP_DEF_REQUIREMENT_T(ItPlusDiff, val<T>() + val<difft_from<T>>());
  CAMP_DEF_REQUIREMENT_T(DiffPlusIt, val<difft_from<T>>() + val<T>());
  CAMP_DEF_REQUIREMENT_T(ItSubEqDiff, val<T>() -= val<difft_from<T>>());
  CAMP_DEF_REQUIREMENT_T(ItSubDiff, val<T>() - val<difft_from<T>>());
  CAMP_DEF_REQUIREMENT_T(DiffSubIt, val<difft_from<T>>() - val<T>());
  CAMP_DEF_REQUIREMENT_T(IndexDiff, val<T>()[val<difft_from<T>>()]);

  CAMP_DEF_CONCEPT_AND_TRAITS_T(random_access_iterator,
                                is_random_access_iterator,
                                CAMP_REQ(bidirectional_iterator, T)
                                    && CAMP_REQ(comparable, T)
                                    && detect_exact<T &, ItPlusEqDiff, T &>()
                                    && detect_exact<T, ItPlusDiff, T>()
                                    && detect_exact<T, DiffPlusIt, T>()
                                    && detect_exact<T &, ItSubEqDiff, T &>()
                                    && detect_exact<T, ItSubDiff, T>()
                                    && detect<IndexDiff, T>());

  CAMP_DEF_REQUIREMENT_T(BeginMember, val<T>().begin());
  CAMP_DEF_REQUIREMENT_T(BeginFree, begin(val<T>()));
  CAMP_DEF_REQUIREMENT_T(EndMember, val<T>().end());
  CAMP_DEF_REQUIREMENT_T(EndFree, end(val<T>()));
  CAMP_DEF_CONCEPT_AND_TRAITS_T(
      has_begin_end,
      trait_has_begin_end,
      (detect<BeginMember, T>() || detect<BeginFree>())
          && (detect<EndMember, T>() || detect<EndFree>()));

  // template <typename T>
  // struct Range : DefineConcept(HasBeginEnd<T>(),
  // Iterator<iterator_from<T>>()) {
  // };
  //
  // template <typename T>
  // struct ForwardRange
  //     : DefineConcept(HasBeginEnd<T>(), ForwardIterator<iterator_from<T>>())
  //     {
  // };
  //
  // template <typename T>
  // struct BidirectionalRange
  //     : DefineConcept(HasBeginEnd<T>(),
  //                     BidirectionalIterator<iterator_from<T>>()) {
  // };

  CAMP_DEF_CONCEPT_AND_TRAITS_T(random_access_range,
                                is_random_access_range,
                                CAMP_REQ(has_begin_end, T)
                                    && CAMP_REQ(random_access_iterator,
                                                iterator_from<T>));


}  // end namespace concepts

namespace type_traits
{

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
    struct IsSpecialized<
        typename void_t<decltype(camp::val<Template<T...>>())>::type,
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
        : camp::concepts::metalib::is_same<Expected<Args...>, Actual<Args...>> {
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
