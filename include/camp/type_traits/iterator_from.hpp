#ifndef __CAMP_TYPE_TRAITS_ITERATOR_FROM
#define __CAMP_TYPE_TRAITS_ITERATOR_FROM

#include "detect.hpp"
#include "enable_if.hpp"

namespace camp
{
namespace detail
{
  namespace iter_from_
  {
    CAMP_DEF_REQUIREMENT_T(BeginMember, val<T>().begin());
    CAMP_DEF_REQUIREMENT_T(BeginFree, begin(val<T>()));
    template <typename T, typename Enable = void>
    struct iterator_from_impl {
    };
    template <typename T>
    struct iterator_from_impl<T, enable_if_t<detect<BeginMember, T>(), void>> {
      using type = decltype(val<T>().begin());
    };
    template <typename T>
    struct iterator_from_impl<T, enable_if_t<(!detect<BeginMember, T>()) && detect<BeginFree, T>(), void>> {
      using type = decltype(begin(val<T>()));
    };
  }  // namespace iter_from_
}  // namespace detail

template <typename T>
using iterator_from = typename detail::iter_from_::iterator_from_impl<T>::type;

}  // namespace camp

#endif /* __CAMP_TYPE_TRAITS_ITERATOR_FROM */
