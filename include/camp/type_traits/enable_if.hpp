#ifndef __CAMP_TYPE_TRAITS_ENABLE_IF
#define __CAMP_TYPE_TRAITS_ENABLE_IF

namespace camp
{

template <bool B, class T = void>
struct enable_if {
};

template <class T>
struct enable_if<true, T> {
  typedef T type;
};

template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

}  // namespace camp

#endif /* __CAMP_TYPE_TRAITS_ENABLE_IF */
