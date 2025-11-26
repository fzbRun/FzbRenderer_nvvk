function(shaders_nvvk_setup)
    set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/_autogen")
    file(GLOB SHADER_GLSL_FILES "*.glsl")
    file(GLOB SHADER_SLANG_FILES "*.slang")

    file(GLOB SHADER_H_FILES "*.h" "*.h.slang")
    list(FILTER SHADER_SLANG_FILES EXCLUDE REGEX ".*\\.h\\.slang$")

    list(APPEND SHADER_SLANG_FILES 
            ${NVSHADERS_DIR}/nvshaders/sky_simple.slang
            ${NVSHADERS_DIR}/nvshaders/tonemapper.slang
        )

    #list(APPEND SHADER_SLANG_FILES ${COMMON_DIR}/shaders/foundation.slang)

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

    source_group("shaders_nvvk" FILES ${SHADER_SLANG_FILES} ${SHADER_GLSL_FILES} ${SHADER_H_FILES})
    source_group("shaders_nvvk/Compiled" FILES ${GENERATED_SHADER_SLANG_HEADERS} ${GENERATED_SHADER_GLSL_HEADERS} ${GENERATED_SHADER_HEADERS})
    # 创建一个虚拟目标，仅用于在 IDE 中显示
    add_custom_target(shaders_nvvk ALL
        SOURCES ${SHADER_SLANG_FILES} ${SHADER_GLSL_FILES} ${SHADER_H_FILES}
                ${GENERATED_SHADER_SLANG_HEADERS} ${GENERATED_SHADER_GLSL_HEADERS}
    )
endfunction()