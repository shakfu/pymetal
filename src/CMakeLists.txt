# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/src/pymetal)
# set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/src/pymetal)

nanobind_add_module(
  # Name of the extension
  _pymetal

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  ${CMAKE_SOURCE_DIR}/src/_pymetal.cpp
)

# Define platform-specific RtAudio macros
target_compile_definitions(_pymetal
    PRIVATE
    $<$<PLATFORM_ID:Darwin>:__MACOSX_CORE__>
    HAVE_CONFIG_H
)

target_include_directories(_pymetal
    PRIVATE
    ${CMAKE_SOURCE_DIR}/thirdparty/metal-cpp
)

target_link_libraries(_pymetal
    PUBLIC
    "$<$<PLATFORM_ID:Darwin>:-framework CoreFoundation>"
    "$<$<PLATFORM_ID:Darwin>:-framework Foundation>"
    "$<$<PLATFORM_ID:Darwin>:-framework Metal>"
)


# set_target_properties(_pymetal PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/src/pymetal"
#     ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/src/pymetal"
#     LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/src/pymetal"
# )


# Install directive for scikit-build-core
install(TARGETS _pymetal LIBRARY DESTINATION pymetal)


