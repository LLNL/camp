//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <sstream>
#include <string>

#include "camp/camp.hpp"
#include "gtest/gtest.h"

struct MyInt {
  int i;
};

namespace camp
{
namespace experimental
{

  template <>
  struct StreamInsertHelper<MyInt&> {
    MyInt& m_val;

    std::ostream& operator()(std::ostream& str) const { return str << m_val.i; }
  };

  template <>
  struct StreamInsertHelper<MyInt const&> {
    MyInt const& m_val;

    std::ostream& operator()(std::ostream& str) const { return str << m_val.i; }
  };

}  // namespace experimental
}  // namespace camp

TEST(CampErrors, Throw)
{
  using namespace std::string_literals;
  ASSERT_THROW(::camp::throw_re(""), std::runtime_error);
  ASSERT_THROW(::camp::throw_re(""s), std::runtime_error);
}

TEST(CampErrors, StreamInsertHelper)
{
  using namespace std::string_literals;

  int i = 7;
  MyInt mi{i};

  std::string expected_string = "7"s;

  {
    std::ostringstream str;
    str << i;
    ASSERT_EQ(str.str(), expected_string);
  }

  {
    std::ostringstream str;
    str << ::camp::experimental::StreamInsertHelper{i};
    ASSERT_EQ(str.str(), expected_string);
  }

  {
    std::ostringstream str;
    str << ::camp::experimental::StreamInsertHelper{static_cast<const int&>(i)};
    ASSERT_EQ(str.str(), expected_string);
  }

  {
    std::ostringstream str;
    str << ::camp::experimental::StreamInsertHelper{std::move(i)};
    ASSERT_EQ(str.str(), expected_string);
  }

  {
    std::ostringstream str;
    str << ::camp::experimental::StreamInsertHelper{mi};
    ASSERT_EQ(str.str(), expected_string);
  }

  {
    std::ostringstream str;
    str << ::camp::experimental::StreamInsertHelper{
        static_cast<const MyInt&>(mi)};
    ASSERT_EQ(str.str(), expected_string);
  }

  {
    std::ostringstream str;
    str << ::camp::experimental::StreamInsertHelper{std::move(mi)};
    ASSERT_EQ(str.str(), expected_string);
  }
}

TEST(CampErrors, InsertArgsString)
{
  using namespace std::string_literals;

  int i = 7;
  MyInt mi{8};
  std::string si = "9"s;

  std::string expected_string_0 = ""s;

  std::string expected_string_1 = "7"s;
  std::string expected_string_1_args = "a=7"s;

  std::string expected_string_3 = "7, 8, 9"s;
  std::string expected_string_3_args = "a=7, bb=8, ccc=9"s;

  {
    std::ostringstream str;
    ::camp::experimental::insertArgsString(str,
                                           "",
                                           std::forward_as_tuple(),
                                           std::make_index_sequence<0>{});
    ASSERT_EQ(str.str(), expected_string_0);
  }

  {
    std::ostringstream str;
    ::camp::experimental::insertArgsString(str,
                                           "",
                                           std::forward_as_tuple(i),
                                           std::make_index_sequence<1>{});
    ASSERT_EQ(str.str(), expected_string_1);
  }

  {
    std::ostringstream str;
    ::camp::experimental::insertArgsString(str,
                                           "a",
                                           std::forward_as_tuple(i),
                                           std::make_index_sequence<1>{});
    ASSERT_EQ(str.str(), expected_string_1_args);
  }

  {
    std::ostringstream str;
    ::camp::experimental::insertArgsString(str,
                                           "",
                                           std::forward_as_tuple(i, mi, si),
                                           std::make_index_sequence<3>{});
    ASSERT_EQ(str.str(), expected_string_3);
  }

  {
    std::ostringstream str;
    ::camp::experimental::insertArgsString(str,
                                           "a bb ccc",
                                           std::forward_as_tuple(i, mi, si),
                                           std::make_index_sequence<3>{});
    ASSERT_EQ(str.str(), expected_string_3_args);
  }
}

TEST(CampErrors, ReportError)
{
  ASSERT_THROW(::camp::reportError("test",
                                   "error occurred",
                                   "func",
                                   "a b c",
                                   std::forward_as_tuple(1, "2", MyInt{3}),
                                   __FILE__,
                                   __LINE__,
                                   true),
               std::runtime_error);
  ASSERT_NO_THROW(::camp::reportError("test",
                                      "error occurred",
                                      "func",
                                      "a b c",
                                      std::forward_as_tuple(1, "2", MyInt{3}),
                                      __FILE__,
                                      __LINE__,
                                      false));
}

TEST(CampErrors, APIInvokeAndCheck)
{
#ifdef CAMP_HAVE_CUDA
  {
    void* ptr = nullptr;
    auto return_success = [](auto...) { return cudaSuccess; };
    auto return_notReady = [](auto...) { return cudaErrorNotReady; };
    auto return_invalid = [](auto...) { return cudaErrorInvalidValue; };

    ASSERT_NO_THROW(CAMP_CUDA_API_INVOKE_AND_CHECK(return_success));
    ASSERT_NO_THROW(CAMP_CUDA_API_INVOKE_AND_CHECK(return_success, 1));
    ASSERT_NO_THROW(CAMP_CUDA_API_INVOKE_AND_CHECK(return_success, 1, ptr));
    ASSERT_EQ(CAMP_CUDA_API_INVOKE_AND_CHECK_RETURN(return_success),
              cudaSuccess);
    ASSERT_EQ(CAMP_CUDA_API_INVOKE_AND_CHECK_RETURN(return_success, 1),
              cudaSuccess);
    ASSERT_EQ(CAMP_CUDA_API_INVOKE_AND_CHECK_RETURN(return_success, 1, ptr),
              cudaSuccess);

    ASSERT_NO_THROW(CAMP_CUDA_API_INVOKE_AND_CHECK(return_notReady));
    ASSERT_NO_THROW(CAMP_CUDA_API_INVOKE_AND_CHECK(return_notReady, 1));
    ASSERT_NO_THROW(CAMP_CUDA_API_INVOKE_AND_CHECK(return_notReady, 1, ptr));
    ASSERT_EQ(CAMP_CUDA_API_INVOKE_AND_CHECK_RETURN(return_notReady),
              cudaErrorNotReady);
    ASSERT_EQ(CAMP_CUDA_API_INVOKE_AND_CHECK_RETURN(return_notReady, 1),
              cudaErrorNotReady);
    ASSERT_EQ(CAMP_CUDA_API_INVOKE_AND_CHECK_RETURN(return_notReady, 1, ptr),
              cudaErrorNotReady);

    ASSERT_THROW(CAMP_CUDA_API_INVOKE_AND_CHECK(return_invalid),
                 std::runtime_error);
    ASSERT_THROW(CAMP_CUDA_API_INVOKE_AND_CHECK(return_invalid, 1),
                 std::runtime_error);
    ASSERT_THROW(CAMP_CUDA_API_INVOKE_AND_CHECK(return_invalid, 1, ptr),
                 std::runtime_error);
    ASSERT_THROW((void)CAMP_CUDA_API_INVOKE_AND_CHECK_RETURN(return_invalid),
                 std::runtime_error);
    ASSERT_THROW((void)CAMP_CUDA_API_INVOKE_AND_CHECK_RETURN(return_invalid, 1),
                 std::runtime_error);
    ASSERT_THROW((void)CAMP_CUDA_API_INVOKE_AND_CHECK_RETURN(return_invalid,
                                                             1,
                                                             ptr),
                 std::runtime_error);
  }
#endif
#ifdef CAMP_HAVE_HIP
  {
    void* ptr = nullptr;
    auto return_success = [](auto...) { return hipSuccess; };
    auto return_notReady = [](auto...) { return hipErrorNotReady; };
    auto return_invalid = [](auto...) { return hipErrorInvalidValue; };

    ASSERT_NO_THROW(CAMP_HIP_API_INVOKE_AND_CHECK(return_success));
    ASSERT_NO_THROW(CAMP_HIP_API_INVOKE_AND_CHECK(return_success, 1));
    ASSERT_NO_THROW(CAMP_HIP_API_INVOKE_AND_CHECK(return_success, 1, ptr));
    ASSERT_EQ(CAMP_HIP_API_INVOKE_AND_CHECK_RETURN(return_success), hipSuccess);
    ASSERT_EQ(CAMP_HIP_API_INVOKE_AND_CHECK_RETURN(return_success, 1),
              hipSuccess);
    ASSERT_EQ(CAMP_HIP_API_INVOKE_AND_CHECK_RETURN(return_success, 1, ptr),
              hipSuccess);

    ASSERT_NO_THROW(CAMP_HIP_API_INVOKE_AND_CHECK(return_notReady));
    ASSERT_NO_THROW(CAMP_HIP_API_INVOKE_AND_CHECK(return_notReady, 1));
    ASSERT_NO_THROW(CAMP_HIP_API_INVOKE_AND_CHECK(return_notReady, 1, ptr));
    ASSERT_EQ(CAMP_HIP_API_INVOKE_AND_CHECK_RETURN(return_notReady),
              hipErrorNotReady);
    ASSERT_EQ(CAMP_HIP_API_INVOKE_AND_CHECK_RETURN(return_notReady, 1),
              hipErrorNotReady);
    ASSERT_EQ(CAMP_HIP_API_INVOKE_AND_CHECK_RETURN(return_notReady, 1, ptr),
              hipErrorNotReady);

    ASSERT_THROW(CAMP_HIP_API_INVOKE_AND_CHECK(return_invalid),
                 std::runtime_error);
    ASSERT_THROW(CAMP_HIP_API_INVOKE_AND_CHECK(return_invalid, 1),
                 std::runtime_error);
    ASSERT_THROW(CAMP_HIP_API_INVOKE_AND_CHECK(return_invalid, 1, ptr),
                 std::runtime_error);
    ASSERT_THROW((void)CAMP_HIP_API_INVOKE_AND_CHECK_RETURN(return_invalid),
                 std::runtime_error);
    ASSERT_THROW((void)CAMP_HIP_API_INVOKE_AND_CHECK_RETURN(return_invalid, 1),
                 std::runtime_error);
    ASSERT_THROW((void)CAMP_HIP_API_INVOKE_AND_CHECK_RETURN(return_invalid,
                                                            1,
                                                            ptr),
                 std::runtime_error);
  }
#endif
}
