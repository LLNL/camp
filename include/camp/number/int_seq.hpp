#ifndef CAMP_INT_SEQ_HPP
#define CAMP_INT_SEQ_HPP

#include "../defines.hpp"

namespace camp
{
// TODO: document
template <typename T, T... vs>
struct int_seq {
  using type = int_seq;
};
/// Index list, use for indexing into parameter packs and lists
template <idx_t... vs>
using idx_seq = int_seq<idx_t, vs...>;
}  // namespace camp

#endif /* CAMP_INT_SEQ_HPP */
