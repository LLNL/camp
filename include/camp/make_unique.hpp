//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __CAMP_make_unique_hpp
#define __CAMP_make_unique_hpp

#include <memory>

namespace camp
{

template <typename T, typename... Args>
constexpr inline std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace camp

#endif /* __CAMP_make_unique_hpp */
