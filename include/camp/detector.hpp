//
// Created by Tom Scogland on 11/5/19.
//

#ifndef CAMP_DETECT_HPP
#define CAMP_DETECT_HPP

#include <type_traits>
#include "number/number.hpp"

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
  using value_t = std::false_type;
  constexpr static auto value = false;
  using type = Default;
};

template <class Default, template <class...> class Concept, class... Args>
struct _detector<Default,
                 typename void_t<Concept<Args...>>::type,
                 Concept,
                 Args...> {
  using value_t = ::std::true_type;
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
constexpr bool is_detected_v =
    detail::_detector<detail::nonesuch, void, Op, Args...>::value;

template <template <class...> class Op, class... Args>
using detected_t = typename is_detected<Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = detail::_detector<Default, void, Op, Args...>;

template <class Default, template <class...> class Op, class... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;


template <class Expected, template <class...> class Op, class... Args>
using is_detected_exact = ::std::is_same<Expected, detected_t<Op, Args...>>;

template <class Expected, template <class...> class Op, class... Args>
constexpr bool is_detected_exact_v =
    ::std::is_same<Expected, detected_t<Op, Args...>>::value;

template <class To, template <class...> class Op, class... Args>
using is_detected_convertible =
    ::std::is_convertible<detected_t<Op, Args...>, To>;

template <class To, template <class...> class Op, class... Args>
constexpr bool is_detected_convertible_v =
    ::std::is_convertible<detected_t<Op, Args...>, To>::value;

}  // namespace camp

#endif
