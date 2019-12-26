//
// Created by Tom Scogland on 11/5/19.
//

#include <camp/detail/test.hpp>
#include <camp/type_traits/is_convertible.hpp>

using namespace camp;

class A {};
class B : public A {};
class C {};
class D { public: operator C() { return c; }  C c; };


CAMP_CHECK_VALUE(is_convertible<B*, A*>);
CAMP_CHECK_VALUE_NOT(is_convertible<A*, B*>);
CAMP_CHECK_VALUE_NOT(is_convertible<B*, C*>);
CAMP_CHECK_VALUE(is_convertible_t<D, C>);

CAMP_CHECK_VALUE(is_convertible<int, int>);
CAMP_CHECK_VALUE(is_convertible<int, double>);
