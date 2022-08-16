
#include <algorithm>
#include <type_traits>
#include <utility>
//
/// meta-type that always produces void
template <class...>
struct void_t {
  using type = void;
};
template <class First, class...>
struct first_t {
  using type = First;
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


#define CAMP_PP_CAT_(X, ...) X##__VA_ARGS__
#define CAMP_PP_CAT(X, ...) CAMP_PP_CAT_(X, __VA_ARGS__)
/* #define CAMP_DECL_REQUIRES(name, args, ...) \ */
/*   auto name args -> decltype((__VA_ARGS), void()){} */
template <typename T>
struct ret_type;
template <typename T, typename... Ts>
struct ret_type<T (*)(Ts...)> {
  using type = T;
};

#if defined(__cpp_concepts) && __cpp_concepts >= 202002L
#define CAMPC_CONCEPT_REQUIRES(name, args, ...) \
  template <typename... Ts>                     \
  concept name = requires args                  \
  {                                             \
    __VA_ARGS__;                                \
  }
#define CAMPC_AND ;
#define CAMPC_CONCEPT concept
#else  // ^^^ concepts ^^^ / vvv no concepts vvv
#define CAMPC_CONCEPT_REQUIRES(name, args, ...)                                \
  auto CAMP_PP_CAT(name, _test_) args->decltype((__VA_ARGS__, void()));        \
  template <typename... Ts>                                                    \
  using CAMP_PP_CAT(name, _detector) =                                         \
      typename ret_type<decltype(&CAMP_PP_CAT(name, _test_) < Ts... >)>::type; \
  template <typename... Ts>                                                    \
  constexpr bool name = is_detected_v<CAMP_PP_CAT(name, _detector), Ts...>
#define CAMPC_AND ,
#define CAMPC_CONCEPT constexpr bool
#endif

/* auto name (tag<Ts...> *,
 * decltype(CAMP_PP_CAT(name,_test_)<Ts...>(std::declval<Ts>()...)) *v=nullptr)
 * {return *v;} \ */
/* auto name (...) -> detail::nonesuch { return {};} */
/* auto name (tag<Ts...> *, decltype(&CAMP_PP_CAT(name,_test_)<Ts...>)
 * v=nullptr) -> decltype( */
template <typename... Ts>
struct tag {
};


namespace _swappable
{
using namespace std;
template <typename T>
CAMPC_CONCEPT_REQUIRES(swappable, (T & a, T &b), swap(a, b));
template <typename T, typename U>
CAMPC_CONCEPT_REQUIRES(swappable_with,
                       (T && a, U &&b),
                       swap(std::forward<T>(a), std::forward<U>(b))
                           CAMPC_AND swap(std::forward<U>(b),
                                          std::forward<T>(a)));
}  // namespace _swappable

using _swappable::swappable;
using _swappable::swappable_with;

template <typename from, typename to>
CAMPC_CONCEPT convertible_to = std::is_convertible<from, to>::value;

/// Constexpr helper for pre-c++20 convertible-to checks
template <typename from, typename to>
constexpr bool convertible_to_c(from&&) {
  return std::is_convertible<from, to>::value;
}

/* template <typename T> */
/* constexpr bool swappable = is_detected_v<swappable_det, T>; */
/* template <class T, std::enable_if_t<swappable<T>, int> = 0> */
/* std::true_type test(); */
/* template <class T, std::enable_if_t<!swappable<T>, int> = 0> */
/* std::false_type test(); */

/* static_assert(!std::is_same<decltype(sd2l_requires((tag<int> *)nullptr)), */
/*                             detail::nonesuch>::value, */
/*               "nodet"); */
#include <concepts>
#include <vector>
#include <ranges>
static_assert(std::swappable<int>, "int swappable");
static_assert(swappable<int>, "int swappable");
void foo(int &a, int &b) {
  std::ranges::swap(a, b);
}
static_assert((bool)std::swappable_with<std::vector<int>&, std::vector<int>&>, "int swappable");
static_assert(swappable_with<std::vector<int>&, std::vector<int>&>, "int swappable");
static_assert(convertible_to<int, ssize_t>, "int swappable");
static_assert(convertible_to<ssize_t, int>, "int swappable");
static_assert(convertible_to<double, int>, "int swappable");
static_assert(convertible_to<double, tag<>>, "int swappable");

/* template <typename T> */
/* using swappable_det = decltype(std::declval<T>() + std::declval<T>()); */

/* template <typename T> */
/* auto sd2l_test(T &a, T &b) -> decltype((a + b, void())) */
/* { */
/* } */

/* template <typename T> */
/* auto sd2l_test_typeret(T &a, T &b) -> decltype(a + b); */

/* auto sd2l_requires(...) -> detail::nonesuch { return {}; } */
/* template <typename... Ts> */
/* auto sd2l_requires(tag<Ts...> *, decltype(&sd2l_test<Ts...>) v = nullptr) */
/* { */
/*   return *v; */
/* } */

/* auto generic_requires(...) -> detail::nonesuch { return {}; } */
/* template <template <typename...> class Op, typename... Ts> */
/* auto generic_requires(tag<Ts...> *, decltype(&sd2l_test<Ts...>) v = nullptr)
 */
/* { */
/*   return *v; */
/* } */

/* template <typename... As> */
/* using sd2l_req = decltype(sd2l_test<As...>); */
/* template <typename... As> */
/* using sd2l_req2 = decltype(sd2l_test_typeret<As...>()); */
/* /1* template<typename ...As> *1/ */
/* /1* auto sd2l_requires(void*, decltype(&sd2l<As...>)=nullptr) -> int; *1/ */

/* template <typename T> */
/* using swappable_det2 = decltype(sd2l(std::declval<T>(), std::declval<T>()));
 */
/* static_assert(is_detected<swappable_det, tag<>>::value, "int swappable"); */
/* static_assert(is_detected<swappable_det, int>::value, "int swappable"); */
/* static_assert(is_detected_convertible<ssize_t, swappable, int>::value, */
/*               "int swappable"); */
/* static_assert(is_detected<sd2l_req, int>::value, "int swappable"); */
/* static_assert(is_detected<sd2l_req2, int>::value, "int swappable"); */
/* static_assert(is_detected<swappable_det, int>::value, "int swappable"); */
/* static_assert(is_detected_v<swappable_det, int>, "int swappable"); */
/* static_assert(is_detected_v<swappable_det2, int>, "int swappable"); */
/* /1* static_assert(test<int>(), "int swappable"); *1/ */
