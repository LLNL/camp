function(camp_add_test TESTNAME)
  cmake_parse_arguments(ABT "GTEST;RUN;OFFLOAD" "" "BUILD;TEST;DEPENDS_ON" ${ARGN})


  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TESTNAME}.cpp")
    list(APPEND ABT_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/${TESTNAME}.cpp")
    list(REMOVE_DUPLICATES ABT_BUILD)
  endif()


  if(NOT CMAKE_BUILD_TYPE)
    set(TEST_BUILD_TYPE Release)
  else()
    set(TEST_BUILD_TYPE ${CMAKE_BUILD_TYPE})
  endif()

  set(_depends camp gtest)

  if(ENABLE_CUDA)
    list(APPEND _depends cuda)
  endif()

  if(ENABLE_HIP)
    list(APPEND _depends hip)
  endif()

  if(ABT_DEPENDS_ON)
    list(APPEND _depends ${ABT_DEPENDS_ON})
  endif()
  # allow masking of offload tests in CI
  if(ABT_OFFLOAD)
    set(TESTNAME ${TESTNAME}.offload)
  endif(ABT_OFFLOAD)


  # Actual build setup
  blt_add_executable(
    NAME ${TESTNAME}
    SOURCES ${ABT_BUILD}
    DEPENDS_ON ${_depends}
    FOLDER test)

  # The test itself
  add_test(NAME ${TESTNAME}
    COMMAND ${TESTNAME}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test
    ${TEST})
endfunction()

camp_add_test(resource GTEST OFFLOAD)
camp_add_test(tuple GTEST)

camp_add_test(accumulate)
camp_add_test(at_key)
camp_add_test(filter)
camp_add_test(find_if)
camp_add_test(flatten)
camp_add_test(index_of)
camp_add_test(lambda)
camp_add_test(number)
camp_add_test(size)
camp_add_test(transform)
