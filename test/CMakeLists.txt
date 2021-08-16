function(camp_add_test TESTNAME)
  cmake_parse_arguments(ABT "GTEST;RUN" "" "BUILD;TEST" ${ARGN})


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
  # target that is run by the test, build and run
  if(ABT_GTEST OR ABT_RUN)
    # Actual build setup, do not build by default
    blt_add_executable(
      NAME ${TESTNAME}
      SOURCES ${ABT_BUILD}
      DEPENDS_ON ${_depends})
    set_target_properties( ${TESTNAME} PROPERTIES EXCLUDE_FROM_ALL On)
    add_custom_target(${TESTNAME}.runner
      COMMAND ${CMAKE_COMMAND} --build . --target ${TESTNAME} --config ${TEST_BUILD_TYPE}
      COMMAND "$<TARGET_FILE:${TESTNAME}>"
      WORKING_DIRECTORY ${CURRENT_BINARY_DIR}
      )
  else()
    # Actual build setup, do not build by default, use a lib so no main required
    blt_add_executable(
      NAME ${TESTNAME}
      SOURCES ${ABT_BUILD}
      DEPENDS_ON ${_depends})
    add_custom_target(${TESTNAME}.runner
      COMMAND ${CMAKE_COMMAND} --build . --target ${TESTNAME} --config ${TEST_BUILD_TYPE}
      WORKING_DIRECTORY ${CURRENT_BINARY_DIR}
      )
  endif()

  if(ENABLE_CUDA)
    set(source_files ${ABT_BUILD})
    list(FILTER source_files INCLUDE REGEX ".*\.cpp$")
    if(NOT ENABLE_CLANG_CUDA)
      set_source_files_properties( ${source_files} PROPERTIES LANGUAGE CUDA )
      set_target_properties( ${TESTNAME} PROPERTIES LINKER_LANGUAGE CUDA)
    else()
      if(BLT_LOADED)
        target_link_libraries(${TESTNAME} cuda)
      else()
        target_link_libraries(${TESTNAME} camp_cuda)
      endif()
    endif()
  endif()


  # The test itself, invoke the runner
  add_test(NAME ${TESTNAME}
    COMMAND ${CMAKE_COMMAND} --build . --target ${TESTNAME}.runner
    WORKING_DIRECTORY ${CURRENT_BINARY_DIR}
    ${TEST})
  set_target_properties(${TESTNAME} PROPERTIES FOLDER test)

  list(GET ABT_BUILD 0 source_file)
  if(NOT source_file)
    return()
  endif()
  set(test_prop_regex  ".*(WILL_FAIL|PASS_REGEX|FAIL_REGEX):?[ ]*(.*)[ ]*")
  file(STRINGS ${source_file} test_props REGEX "${test_prop_regex}")
  foreach(line ${test_props})

    if(NOT line MATCHES "${test_prop_regex}")
      continue()
    endif()
    if(CMAKE_MATCH_1 STREQUAL "WILL_FAIL")
      set_tests_properties(${TESTNAME} PROPERTIES WILL_FAIL)
    elseif(CMAKE_MATCH_1 STREQUAL "PASS_REGEX")
      set_tests_properties(${TESTNAME} PROPERTIES
        PASS_REGULAR_EXPRESSION "${CMAKE_MATCH_2}")
    elseif(CMAKE_MATCH_1 STREQUAL "FAIL_REGEX")
      set_tests_properties(${TESTNAME} PROPERTIES
        FAIL_REGULAR_EXPRESSION "${CMAKE_MATCH_2}")
    endif()
  endforeach()
endfunction()

camp_add_test(resource GTEST)
camp_add_test(tuple GTEST)

camp_add_test(tuple_out_of_range RUN)

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
