//
// Created by Tom Scogland on 11/5/19.
//

#ifndef CAMP_DETECT_HPP
#define CAMP_DETECT_HPP

#include "../number/number.hpp"
#include "../type_traits/is_convertible.hpp"
#include "../type_traits/is_same.hpp"

namespace camp
{

namespace detail
{
  template <class...>
  struct void_t {
    using type = void;
  };

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

  template <typename... T>
  camp::true_type ___valid_expr___(T &&...) noexcept;
}  // namespace detail

template <template <class...> class Op, class... Args>
using is_detected = detail::_detector<detail::nonesuch, void, Op, Args...>;

template <template <class...> class Op, class... Args>
using detected_t = typename is_detected<Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = detail::_detector<Default, void, Op, Args...>;

template <class Default, template <class...> class Op, class... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;


template <class Expected, template <class...> class Op, class... Args>
using is_detected_exact = is_same<Expected, detected_t<Op, Args...>>;

template <class To, template <class...> class Op, class... Args>
using is_detected_convertible = is_convertible<detected_t<Op, Args...>, To>;


#define CAMP_DEF_DETECTOR(name, params, ...)                   \
  template <CAMP_UNQUOTE params, typename __Res = __VA_ARGS__> \
  using name = __Res

#define CAMP_DEF_DETECTOR_T(name, ...)                   \
  template <class T, typename __Res = __VA_ARGS__> \
  using name = __Res

#define CAMP_DEF_DETECTOR_TU(name, ...)                   \
  template <class T, class U, typename __Res = __VA_ARGS__> \
  using name = __Res

#define CAMP_DEF_CONCEPT(name, params, ...) \
  CAMP_DEF_DETECTOR(name,                   \
                    params,                 \
                    decltype(detail::___valid_expr___(__VA_ARGS__)))

}  // namespace camp
#endif  // CAMP_DETECT_HPP
