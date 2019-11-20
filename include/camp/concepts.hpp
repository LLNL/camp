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

#include <type_traits>

#include "helpers.hpp"
#include "list.hpp"
#include "number.hpp"

#include "type_traits/detect.hpp"
#include "type_traits/enable_if.hpp"
#include "type_traits/is_same.hpp"
#include "type_traits/iterator_from.hpp"

namespace camp
{
namespace concepts
{
  namespace internal
  {
    CAMP_DEF_REQUIREMENT_TU(LessThan, val<T>() < val<U>());
    CAMP_DEF_REQUIREMENT_TU(GreaterThan, val<T>() > val<U>());
    CAMP_DEF_REQUIREMENT_TU(LessEqual, val<T>() <= val<U>());
    CAMP_DEF_REQUIREMENT_TU(GreaterEqual, val<T>() >= val<U>());

    CAMP_DEF_REQUIREMENT_TU(Equality, val<T>() == val<U>());
    CAMP_DEF_REQUIREMENT_TU(Inequality, val<T>() != val<U>());

    CAMP_DEF_CONCEPT_TU(__Weakly_equality_comparable_with,
                        detect_convertible<bool, Equality, T, U>()
                            && detect_convertible<bool, Inequality, T, U>()
                            && detect_convertible<bool, Equality, U, T>()
                            && detect_convertible<bool, Inequality, U, T>());

  }  // namespace internal

  CAMP_DEF_CONCEPT_T(equality_comparable,
                     CAMP_REQ(internal::__Weakly_equality_comparable_with,
                              T,
                              T));

  CAMP_DEF_CONCEPT_TU(
      equality_comparable_with,
      CAMP_REQ(equality_comparable, T) && CAMP_REQ(equality_comparable, U)
          && CAMP_REQ(internal::__Weakly_equality_comparable_with, T, U));

  CAMP_DEF_CONCEPT_TU(
      comparable_with,
      CAMP_REQ(equality_comparable_with, T, U)
          && detect_convertible<bool, internal::LessThan, T, U>()
          && detect_convertible<bool, internal::GreaterThan, T, U>()
          && detect_convertible<bool, internal::LessEqual, T, U>()
          && detect_convertible<bool, internal::GreaterEqual, T, U>()
          && detect_convertible<bool, internal::LessThan, U, T>()
          && detect_convertible<bool, internal::GreaterThan, U, T>()
          && detect_convertible<bool, internal::LessEqual, U, T>()
          && detect_convertible<bool, internal::GreaterEqual, U, T>());

  CAMP_DEF_CONCEPT_T(comparable, CAMP_REQ(comparable_with, T, T));

  CAMP_DEF_CONCEPT_T(arithmetic, std::is_arithmetic<T>::value);
  CAMP_DEF_CONCEPT_T(floating_point, std::is_floating_point<T>::value);
  CAMP_DEF_CONCEPT_T(integral, std::is_integral<T>::value);
  CAMP_DEF_CONCEPT_T(signed_, std::is_signed<T>::value);
  CAMP_DEF_CONCEPT_T(unsigned_, std::is_unsigned<T>::value);

  CAMP_DEF_REQUIREMENT_T(Dereference, *(val<T>()));
  CAMP_DEF_REQUIREMENT_T(IncrementPre, ++val<T>());
  CAMP_DEF_CONCEPT_T(iterator,
                     detect<Dereference, T>()
                         && detect_exact<T &, IncrementPre, T &>());

  CAMP_DEF_REQUIREMENT_T(IncrementPost, val<T>()++);
  CAMP_DEF_REQUIREMENT_T(DereferenceIncrementPost, *val<T>()++);
  CAMP_DEF_CONCEPT_T(forward_iterator,
                     CAMP_REQ(iterator, T) && detect<IncrementPost, T &>()
                         && detect<DereferenceIncrementPost, T &>());

  CAMP_DEF_REQUIREMENT_T(DecrementPre, --val<T>());
  CAMP_DEF_REQUIREMENT_T(DecrementPost, val<T>()--);
  CAMP_DEF_REQUIREMENT_T(DereferenceDecrementPost, *val<T>()--);
  CAMP_DEF_CONCEPT_T(bidirectional_iterator,
                     CAMP_REQ(forward_iterator, T)
                         && detect_exact<T &, DecrementPre, T &>()
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

  CAMP_DEF_CONCEPT_T(random_access_iterator,
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
  CAMP_DEF_CONCEPT_T(has_begin_end,
                     (detect<BeginMember, T>() || detect<BeginFree>())
                         && (detect<EndMember, T>() || detect<EndFree>()));

  CAMP_DEF_CONCEPT_T(range,
                     CAMP_REQ(has_begin_end, T)
                         && CAMP_REQ(iterator, iterator_from<T>));
  CAMP_DEF_CONCEPT_T(random_access_range,
                     CAMP_REQ(has_begin_end, T)
                         && CAMP_REQ(random_access_iterator, iterator_from<T>));

  namespace internal
  {
    template <typename Fn, typename Void = void, typename... Args>
    struct invokable : false_type {
      using ret = detail::nonesuch;
    };
    template <typename Fn, typename... Args>
    struct invokable<Fn,
                     typename void_t<decltype(val<Fn>()(val<Args>()...))>::type,
                     Args...> : true_type {
      using ret = decltype(val<Fn>()(val<Args>()...));
    };
  }  // namespace internal

  template <typename Fn, typename... Args>
  using invokable = internal::invokable<Fn, void, Args...>;

  template <typename Fn, typename Ret, typename... Args>
  using invokable_returns =
      is_same_t<typename internal::invokable<Fn, void, Args...>::ret, Ret>;


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
        : camp::is_same<Expected<Args...>, Actual<Args...>> {
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
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::equality_comparable,
                                is_equality_comparable);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::equality_comparable_with,
                                is_equality_comparable_with);

  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::comparable_with, is_comparable_with);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::comparable, is_comparable);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::arithmetic, is_arithmetic);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::floating_point, is_floating_point);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::integral, is_integral);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::signed_, is_signed);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::unsigned_, is_unsigned);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::iterator, is_iterator);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::forward_iterator,
                                is_forward_iterator);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::bidirectional_iterator,
                                is_bidirectional_iterator);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::random_access_iterator,
                                is_random_access_iterator);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::has_begin_end, has_begin_end);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::range,
                                is_range);
  CAMP_TYPE_TRAITS_FROM_CONCEPT(concepts::random_access_range,
                                is_random_access_range);

}  // end namespace type_traits
}  // namespace camp

#endif /* CAMP_CONCEPTS_HPP */
