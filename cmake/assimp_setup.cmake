# 定义一个查找Assimp库的函数
# 用法: find_assimp_library([ROOT_DIR <path>] [DEFAULT_PATH <path>])
function(find_assimp_library)
    set(options "")
    set(oneValueArgs ROOT_DIR DEFAULT_PATH)
    set(multiValueArgs "")
    cmake_parse_arguments(FIND_ASSIMP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # 清理之前的变量
    unset(assimp_INCLUDE_DIR CACHE)
    unset(assimp_DEBUG_LIB_DIR CACHE)
    unset(assimp_DEBUG_DLL_DIR CACHE)
    unset(assimp_RELEASE_LIB_DIR CACHE)
    unset(assimp_RELEASE_DLL_DIR CACHE)
    unset(assimp_FOUND CACHE)
    
    # 确定搜索路径
    set(SEARCH_PATHS)
    if(FIND_ASSIMP_ROOT_DIR)
        list(APPEND SEARCH_PATHS "${FIND_ASSIMP_ROOT_DIR}")
    endif()
    
    if(FIND_ASSIMP_DEFAULT_PATH)
        list(APPEND SEARCH_PATHS "${FIND_ASSIMP_DEFAULT_PATH}")
    endif()
    
    # 添加一些常见的默认路径（可根据需要修改）
    list(APPEND SEARCH_PATHS
        "${CMAKE_SOURCE_DIR}/third_party/assimp"
    )
    
    # 遍历所有搜索路径
    foreach(ASSIMP_ROOT IN LISTS SEARCH_PATHS)
        if(NOT ASSIMP_ROOT)
            continue()
        endif()
        
        message(STATUS "Checking for assimp at: ${ASSIMP_ROOT}")
        
        # 检查头文件
        if(EXISTS "${ASSIMP_ROOT}/include/assimp/aabb.h")
            set(found_include TRUE)
            set(assimp_INCLUDE_DIR "${ASSIMP_ROOT}/include")
            message(STATUS "  Found include directory: ${assimp_INCLUDE_DIR}")
        else()
            set(found_include FALSE)
            message(STATUS "  aabb.h not found in ${ASSIMP_ROOT}/include/assimp/")
        endif()
        
        # 检查Debug库文件
        set(debug_lib_found FALSE)
        set(release_lib_found FALSE)
        set(debug_dll_found FALSE)
        set(release_dll_found FALSE)
        
        # 尝试多种可能的库文件位置和命名模式
        set(DEBUG_LIB_PATHS
            "${ASSIMP_ROOT}/build/lib/Debug/assimp-vc143-mtd.lib"
            "${ASSIMP_ROOT}/build/lib/Debug/assimp-vc142-mtd.lib"
            "${ASSIMP_ROOT}/build/lib/Debug/assimp-vc141-mtd.lib"

            "${ASSIMP_ROOT}/build/lib/Debug/assimp-vc143-mdd.lib"
            "${ASSIMP_ROOT}/build/lib/Debug/assimp-vc142-mdd.lib"
            "${ASSIMP_ROOT}/build/lib/Debug/assimp-vc141-mdd.lib"
        )
        
        set(RELEASE_LIB_PATHS
            "${ASSIMP_ROOT}/build/lib/Release/assimp-vc143-mt.lib"
            "${ASSIMP_ROOT}/build/lib/Release/assimp-vc142-mt.lib"
            "${ASSIMP_ROOT}/build/lib/Release/assimp-vc141-mt.lib"

            "${ASSIMP_ROOT}/build/lib/Release/assimp-vc143-md.lib"
            "${ASSIMP_ROOT}/build/lib/Release/assimp-vc142-md.lib"
            "${ASSIMP_ROOT}/build/lib/Release/assimp-vc141-md.lib"
        )
        
        set(DEBUG_DLL_PATHS
            "${ASSIMP_ROOT}/build/bin/Debug/assimp-vc143-mtd.dll"
            "${ASSIMP_ROOT}/build/bin/Debug/assimp-vc142-mtd.dll"
            "${ASSIMP_ROOT}/build/bin/Debug/assimp-vc141-mtd.dll"

            "${ASSIMP_ROOT}/build/bin/Debug/assimp-vc143-mdd.dll"
            "${ASSIMP_ROOT}/build/bin/Debug/assimp-vc142-mdd.dll"
            "${ASSIMP_ROOT}/build/bin/Debug/assimp-vc141-mdd.dll"
        )
        
        set(RELEASE_DLL_PATHS
            "${ASSIMP_ROOT}/build/bin/Release/assimp-vc143-mt.dll"
            "${ASSIMP_ROOT}/build/bin/Release/assimp-vc142-mt.dll"
            "${ASSIMP_ROOT}/build/bin/Release/assimp-vc141-mt.dll"

            "${ASSIMP_ROOT}/build/bin/Release/assimp-vc143-md.dll"
            "${ASSIMP_ROOT}/build/bin/Release/assimp-vc142-md.dll"
            "${ASSIMP_ROOT}/build/bin/Release/assimp-vc141-md.dll"
        )
        
        # 查找Debug库
        foreach(lib_path IN LISTS DEBUG_LIB_PATHS)
            if(EXISTS "${lib_path}")
                set(assimp_DEBUG_LIB_DIR "${lib_path}")
                set(debug_lib_found TRUE)
                message(STATUS "  Found debug library: ${lib_path}")
                break()
            endif()
        endforeach()
        
        # 查找Release库
        foreach(lib_path IN LISTS RELEASE_LIB_PATHS)
            if(EXISTS "${lib_path}")
                set(assimp_RELEASE_LIB_DIR "${lib_path}")
                set(release_lib_found TRUE)
                message(STATUS "  Found release library: ${lib_path}")
                break()
            endif()
        endforeach()
        
        # 查找Debug DLL
        foreach(dll_path IN LISTS DEBUG_DLL_PATHS)
            if(EXISTS "${dll_path}")
                set(assimp_DEBUG_DLL_DIR "${dll_path}")
                set(debug_dll_found TRUE)
                message(STATUS "  Found debug dll: ${dll_path}")
                break()
            endif()
        endforeach()
        
        # 查找Release DLL
        foreach(dll_path IN LISTS RELEASE_DLL_PATHS)
            if(EXISTS "${dll_path}")
                set(assimp_RELEASE_DLL_DIR "${dll_path}")
                set(release_dll_found TRUE)
                message(STATUS "  Found release dll: ${dll_path}")
                break()
            endif()
        endforeach()
        
        # 如果找到足够的组件，设置found标志并退出循环
        if(found_include AND (debug_lib_found OR release_lib_found))
            set(assimp_ROOT "${ASSIMP_ROOT}" PARENT_SCOPE)
            set(assimp_INCLUDE_DIR "${assimp_INCLUDE_DIR}" PARENT_SCOPE)
            set(assimp_DEBUG_LIB_DIR "${assimp_DEBUG_LIB_DIR}" PARENT_SCOPE)
            set(assimp_DEBUG_DLL_DIR "${assimp_DEBUG_DLL_DIR}" PARENT_SCOPE)
            set(assimp_RELEASE_LIB_DIR "${assimp_RELEASE_LIB_DIR}" PARENT_SCOPE)
            set(assimp_RELEASE_DLL_DIR "${assimp_RELEASE_DLL_DIR}" PARENT_SCOPE)
            set(assimp_FOUND TRUE PARENT_SCOPE)
            message(STATUS "Assimp found successfully at: ${ASSIMP_ROOT}")
            return()
        endif()
    endforeach()
    
    # 如果没找到，设置not found
    message(STATUS "Assimp not found in any of the search paths")
    set(assimp_FOUND FALSE PARENT_SCOPE)
endfunction()

# 创建一个简化的目标创建函数
function(setup_assimp_target TARGET_NAME)
    # 查找Assimp库
    find_assimp_library(
        ROOT_DIR "${assimp_ROOT}"
        #DEFAULT_PATH "${CMAKE_SOURCE_DIR}/third_party/assimp"
    )
    
    if(NOT assimp_FOUND)
        message("Assimp library auto download from github.")

        include(FetchContent)
        FetchContent_Declare(assimp
            GIT_REPOSITORY https://github.com/assimp/assimp.git
            GIT_TAG master
            SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/assimp
            BINARY_DIR ${CMAKE_SOURCE_DIR}/third_party/assimp/build
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DBUILD_SHARED_LIBS=OFF
                -DASSIMP_BUILD_TESTS=OFF
                -DASSIMP_INJECT_DEBUG_POSTFIX=OFF
                -DASSIMP_INSTALL=OFF
                -DASSIMP_BUILD_ASSIMP_TOOLS=OFF
                -DASSIMP_BUILD_ZLIB=ON  # 通常需要
        )
        FetchContent_MakeAvailable(assimp)

        set(assimp_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/assimp/include)
        set(assimp_DEBUG_LIB_DIR assimp)
        set(assimp_RELEASE_LIB_DIR assimp)
    endif()
    
    # 设置包含目录
    target_include_directories(${TARGET_NAME} PRIVATE ${assimp_INCLUDE_DIR})
    target_include_directories(${TARGET_NAME} PRIVATE ${assimp_ROOT}/build/include)
    
    # 根据构建类型链接库
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        if(assimp_DEBUG_LIB_DIR)
            target_link_libraries(${TARGET_NAME} PRIVATE "${assimp_DEBUG_LIB_DIR}")
        else()
            target_link_libraries(${TARGET_NAME} PRIVATE "${assimp_RELEASE_LIB_DIR}")
        endif()
    else()
        if(assimp_RELEASE_LIB_DIR)
            target_link_libraries(${TARGET_NAME} PRIVATE "${assimp_RELEASE_LIB_DIR}")
        else()
            target_link_libraries(${TARGET_NAME} PRIVATE "${assimp_DEBUG_LIB_DIR}")
        endif()
    endif()
    
    # 复制DLL文件到输出目录（Windows下）
    if(WIN32)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND assimp_DEBUG_DLL_DIR)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${assimp_DEBUG_DLL_DIR}"
                $<TARGET_FILE_DIR:${TARGET_NAME}>
            )
        elseif(assimp_RELEASE_DLL_DIR)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${assimp_RELEASE_DLL_DIR}"
                $<TARGET_FILE_DIR:${TARGET_NAME}>
            )
        endif()
    endif()

endfunction()

setup_assimp_target(${PROJECT_NAME})
