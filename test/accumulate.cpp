#include <camp/camp.hpp>

using namespace camp;
CAMP_CHECK_TSAME((accumulate<append, list<>, list<int, float, double>>),
                 (list<int, float, double>));
