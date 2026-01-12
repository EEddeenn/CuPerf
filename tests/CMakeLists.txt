if(CUPERF_ENABLE_TESTS)
  include(FetchContent)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(test_statistics
    test_statistics.cpp
    ../src/core/Statistics.cpp
  )

  target_link_libraries(test_statistics PRIVATE
    GTest::gtest_main
    fmt::fmt
  )
  target_include_directories(test_statistics PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
  )

  add_executable(test_types
    test_types.cpp
    ../src/core/Types.cpp
  )

  target_link_libraries(test_types PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    fmt::fmt
  )
  target_include_directories(test_types PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
  )

  add_test(NAME test_statistics COMMAND test_statistics)
  add_test(NAME test_types COMMAND test_types)
endif()
