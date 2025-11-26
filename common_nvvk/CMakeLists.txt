set(LIB_NAME common_nvvk)

file(GLOB LIB_SOURCES *.cpp *.hpp)
file(GLOB SHADERS_SOURCES shaders/*.*)

# Adding the library here does not actually create a build target,
add_library(${LIB_NAME})
# Add the sources to the library target
target_sources(${LIB_NAME} PRIVATE ${LIB_SOURCES} ${SHADERS_SOURCES})

# Add the vk_mini_samples directory to the include path, so that these files
# can be included using a "common/" prefix.
target_include_directories(${LIB_NAME} INTERFACE ${PROJECT_SOURCE_DIR})

# Add the current directory to include path for the library
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add dependencies
target_link_libraries(${LIB_NAME} PUBLIC 
  glm
  volk # The Vulkan loader
  nvutils
  nvvk
)

# Since this is actually header-only, we don't need to build anything
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
