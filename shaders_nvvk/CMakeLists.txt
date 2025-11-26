set(LIB_NAME shaders_nvvk)
add_library(${LIB_NAME})

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/_autogen")
file(GLOB SHADER_compile_FILES "${SHADER_OUTPUT_DIR}/*.*")

file(GLOB SHADER_GLSL_FILES "*.glsl")
file(GLOB SHADER_SLANG_FILES "*.slang")

file(GLOB SHADER_H_FILES "*.h" "*.h.slang")
list(FILTER SHADER_SLANG_FILES EXCLUDE REGEX ".*\\.h\\.slang$")

list(APPEND SHADER_SLANG_FILES 
        ${NVSHADERS_DIR}/nvshaders/sky_simple.slang
        ${NVSHADERS_DIR}/nvshaders/tonemapper.slang
    )
list(APPEND SHADER_SLANG_FILES ${COMMON_DIR}/shaders/foundation.slang)

set(SHADER_INCLUDE_FLAGS "-I${NVSHADERS_DIR}" "-I${ROOT_DIR}")


compile_slang(
    "${SHADER_SLANG_FILES}"
    "${SHADER_OUTPUT_DIR}"
    GENERATED_SHADER_HEADERS
    EXTRA_FLAGS ${SHADER_INCLUDE_FLAGS}
)
compile_glsl(
    "${SHADER_GLSL_FILES}"
    "${SHADER_OUTPUT_DIR}"
    GENERATED_SHADER_GLSL_HEADERS
    EXTRA_FLAGS ${SHADER_INCLUDE_FLAGS}
)

list(REMOVE_ITEM SHADER_SLANG_FILES "${COMMON_DIR}/shaders/foundation.slang")

target_sources(${LIB_NAME} PRIVATE
  ${SHADER_SLANG_FILES} ${GENERATED_SHADER_SLANG_HEADERS} ${GENERATED_SHADER_GLSL_HEADERS} ${SHADER_H_FILES} 
  ${SHADER_compile_FILES})
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

source_group("shaders_nvvk" FILES ${SHADER_SLANG_FILES} ${SHADER_GLSL_FILES} ${SHADER_H_FILES})
source_group("shaders_nvvk/_autogen" FILES ${SHADER_compile_FILES})
