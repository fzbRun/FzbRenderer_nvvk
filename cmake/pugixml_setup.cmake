# Step 1: 用户可指定 pugixml_ROOT（预期是源码根目录）
if(pugixml_ROOT AND EXISTS "${pugixml_ROOT}")
    # 检查常见布局：pugixml.hpp 是否在 src/ 或根目录
    if(EXISTS "${pugixml_ROOT}/src/pugixml.hpp")
        set(pugixml_INCLUDE_DIR "${pugixml_ROOT}/src")
        message(STATUS "Using user-provided pugixml at: ${pugixml_ROOT} (src layout)")
    elseif(EXISTS "${pugixml_ROOT}/pugixml.hpp")
        set(pugixml_INCLUDE_DIR "${pugixml_ROOT}")
        message(STATUS "Using user-provided pugixml at: ${pugixml_ROOT} (flat layout)")
    else()
        message(WARNING "pugixml_ROOT=${pugixml_ROOT} provided, but pugixml.hpp not found in expected locations.")
        unset(pugixml_ROOT)
    endif()
endif()

# Step 2: 如果还没找到，尝试从已存在的 third_party 目录加载
if(NOT pugixml_INCLUDE_DIR)
    set(PUGIXML_LOCAL_PATH "${CMAKE_SOURCE_DIR}/third_party/pugixml")
    if(EXISTS "${PUGIXML_LOCAL_PATH}/src/pugixml.hpp")
        set(pugixml_INCLUDE_DIR "${PUGIXML_LOCAL_PATH}/src")
        message(STATUS "Found local pugixml in third_party: ${pugixml_INCLUDE_DIR}")
    endif()
endif()

# Step 3: 如果仍然没找到，从 GitHub 克隆
if(NOT pugixml_INCLUDE_DIR)
    message(STATUS "pugixml not found. Cloning from GitHub into third_party...")

    set(PUGIXML_CLONE_DIR "${CMAKE_SOURCE_DIR}/third_party/pugixml")
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party")

    if(NOT EXISTS "${PUGIXML_CLONE_DIR}/src/pugixml.hpp")
        find_package(Git REQUIRED QUIET)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch v1.14 https://github.com/zeux/pugixml.git pugixml
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party"
            RESULT_VARIABLE GIT_CLONE_RESULT
            ERROR_VARIABLE GIT_CLONE_ERROR
            OUTPUT_QUIET
        )
        if(NOT GIT_CLONE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to clone pugixml: ${GIT_CLONE_ERROR}")
        endif()
    endif()

    set(pugixml_INCLUDE_DIR "${PUGIXML_CLONE_DIR}/src")
    message(STATUS "Using downloaded pugixml from: ${pugixml_INCLUDE_DIR}")
endif()

# Step 4: 创建 INTERFACE 库供使用
add_library(pugixml INTERFACE)
target_include_directories(pugixml INTERFACE "${pugixml_INCLUDE_DIR}")
target_compile_definitions(pugixml INTERFACE PUGIXML_HEADER_ONLY)