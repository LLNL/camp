//
// Created by Tom Scogland on 11/5/19.
//

#ifndef CAMP_DETECT_HPP
#define CAMP_DETECT_HPP

#include "../number/number.hpp"
#include "is_convertible.hpp"
#include "is_same.hpp"

namespace camp
{

/// meta-type that always produces void
template <class...>
struct void_t {
  using type = void;
};

namespace detail
{

  struct nonesuch {

    nonesuch() = delete;
    ~nonesuch() = delete;
    nonesuch(nonesuch const &) = delete;
    void operator=(nonesuch const &) = delete;
  };

  template <class Default,
            class /* always void*/,
            template <class...>
            class Concept,
            class... Args>
  struct _detector {
    using value_t = ::camp::false_type;
    constexpr static auto value = false;
    using type = Default;
  };

  template <class Default, template <class...> class Concept, class... Args>
  struct _detector<Default,
                   typename void_t<Concept<Args...>>::type,
                   Concept,
                   Args...> {
    using value_t = ::camp::true_type;
    constexpr static auto value = true;
    //    using type = typename Concept<Args...>::type;
    using type = Concept<Args...>;
  };

}  // namespace detail

/// Detect whether a given template/alias Op can be expanded to a valid type
/// expression with Args...
template <template <class...> class Op, class... Args>
using is_detected = detail::_detector<detail::nonesuch, void, Op, Args...>;

template <template <class...> class Op, class... Args>
constexpr bool detect()
{
  return is_detected<Op, Args...>::value;
}

template <template <class...> class Op, class... Args>
using detected_t = typename is_detected<Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = detail::_detector<Default, void, Op, Args...>;

template <class Default, template <class...> class Op, class... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;


template <class Expected, template <class...> class Op, class... Args>
using is_detected_exact = is_same<Expected, detected_t<Op, Args...>>;

template <class Expected, template <class...> class Op, class... Args>
constexpr bool detect_exact()
{
  return is_detected_exact<Expected, Op, Args...>::value;
}

template <class To, template <class...> class Op, class... Args>
using is_detected_convertible = is_convertible<detected_t<Op, Args...>, To>;

template <class To, template <class...> class Op, class... Args>
constexpr bool detect_convertible()
{
  return is_detected_convertible<To, Op, Args...>::value;
}


#define CAMP_DEF_DETECTOR(name, params, ...)                   \
  template <CAMP_UNQUOTE params, typename __Res = __VA_ARGS__> \
  using name = __Res

#define CAMP_DEF_DETECTOR_T(name, ...)             \
  template <class T, typename __Res = __VA_ARGS__> \
  using name = __Res

#define CAMP_DEF_DETECTOR_TU(name, ...)                     \
  template <class T, class U, typename __Res = __VA_ARGS__> \
  using name = __Res

#define CAMP_DEF_REQUIREMENT(name, params, ...) \
  CAMP_DEF_DETECTOR(name, params, decltype((__VA_ARGS__)))

#define CAMP_DEF_REQUIREMENT_T(name, ...) \
  CAMP_DEF_REQUIREMENT(name, (class T), __VA_ARGS__)

#define CAMP_DEF_REQUIREMENT_TU(name, ...) \
  CAMP_DEF_REQUIREMENT(name, (class T, class U), __VA_ARGS__)


#ifdef HAVE_CONCEPTS
#define CAMP_DEF_CONCEPT(name, params, ...) \
  template <CAMP_UNQUOTE params>            \
  concept name = __VA_ARGS__

#define CAMP_REQ(name, ...) name<__VA_ARGS__>
#elif defined(__cpp_variable_templates) && __cpp_variable_templates >= 201304
#define CAMP_DEF_CONCEPT(name, params, ...) \
  template <CAMP_UNQUOTE params>            \
  CAMP_INLINE_VARIABLE constexpr bool name = __VA_ARGS__
#define CAMP_REQ(name, ...) name<__VA_ARGS__>
#else
#define CAMP_DEF_CONCEPT(name, params, ...) \
  template <CAMP_UNQUOTE params>            \
  constexpr bool name##_c()                 \
  {                                         \
    return (__VA_ARGS__);                   \
  }
#define CAMP_REQ(name, ...) name##_c<__VA_ARGS__>()
#endif

#define CAMP_DEF_CONCEPT_T(name, ...) \
  CAMP_DEF_CONCEPT(name, (class T), __VA_ARGS__)
#define CAMP_DEF_CONCEPT_TU(name, ...) \
  CAMP_DEF_CONCEPT(name, (class T, class U), __VA_ARGS__)

#if defined(CAMP_HAS_VARIABLE_TEMPLATES)
#define CAMP_MAKE_V_FROM_CONCEPT(CONCEPT, TT) \
  template <typename... Types>                \
  constexpr auto TT##_v = CAMP_REQ(CONCEPT, Types...)
#else
#define CAMP_MAKE_V_FROM_CONCEPT(X, Y)
#endif


#define CAMP_TYPE_TRAITS_FROM_CONCEPT(CONCEPT, TT)           \
  template <typename... Types>                               \
  struct TT : num<CAMP_REQ(CONCEPT, Types...)> { \
  };                                                         \
  template <typename... Types>                               \
  using TT##_t = typename TT<Types...>::type;          \
  CAMP_MAKE_V_FROM_CONCEPT(CONCEPT, TT)

#define CAMP_DEF_CONCEPT_AND_TRAITS(CONCEPT, TT, params, ...) \
  CAMP_DEF_CONCEPT(CONCEPT, params, __VA_ARGS__);             \
  CAMP_TYPE_TRAITS_FROM_CONCEPT(CONCEPT, TT)

#define CAMP_DEF_CONCEPT_AND_TRAITS_T(CONCEPT, TT, ...) \
  CAMP_DEF_CONCEPT_T(CONCEPT, __VA_ARGS__);             \
  CAMP_TYPE_TRAITS_FROM_CONCEPT(CONCEPT, TT)

#define CAMP_DEF_CONCEPT_AND_TRAITS_TU(CONCEPT, TT, ...) \
  CAMP_DEF_CONCEPT_TU(CONCEPT, __VA_ARGS__);             \
  CAMP_TYPE_TRAITS_FROM_CONCEPT(CONCEPT, TT)

}  // namespace camp
#endif  // CAMP_DETECT_HPP
