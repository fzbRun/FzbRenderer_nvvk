cmake_minimum_required(VERSION 3.22)

get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(
    ${PROJECT_NAME}
    VERSION 1.0.0
)
#--------------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TUTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common_nvvk")
#------------------------------------------------------------------------------------------------------
file(GLOB_RECURSE EXECUTABLE_SOURCES 
    CONFIGURE_DEPENDS
    src/*.*
    # src/*.h
    # src/*.hpp
    # src/*.cpp
    # src/*.cuh
    # src/*.cu
)

add_executable(${PROJECT_NAME} ${EXECUTABLE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${EXECUTABLE_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE
    nvpro2::nvapp
    nvpro2::nvgui
    nvpro2::nvslang
    nvpro2::nvutils
    nvpro2::nvvk
    nvpro2::nvshaders_host
    nvpro2::nvaftermath
    nvpro2::nvvkgltf
    nvpro2::nvvkglsl
)

add_project_definitions(${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR} ${ROOT_DIR})
#------------------------------------------------------------------------------------------------------
#add_subdirectory(common_nvvk)
#------------------------------------------------------------------------------------------------------
#add_subdirectory(shaders_nvvk)
#------------------------------------------------------------------------------------------------------
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/pugixml_setup.cmake")
target_link_libraries(${PROJECT_NAME} PRIVATE pugixml)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/assimp_setup.cmake")
#--------------------------------------------------------------------------------------
set(COPY_FILES ${NsightAftermath_DLLS})
list(APPEND COPY_FILES ${EXTRA_COPY_FILES})

set(COPY_DIRECTORIES)
list(APPEND COPY_DIRECTORIES ${EXTRA_COPY_DIRECTORIES})

copy_to_runtime_and_install(${PROJECT_NAME}
    FILES ${COPY_FILES} ${Slang_GLSLANG}
    DIRECTORIES ${COPY_DIRECTORIES}
    AUTO
)
#------------------------------------------------------------------------------------------------------
file(GLOB_RECURSE RESOURCES_XML_FILES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/*.xml
)

file(GLOB_RECURSE RENDERERINFO_XML_FILES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/rendererInfo/*.xml
)

target_sources(${PROJECT_NAME} PRIVATE
    ${RESOURCES_XML_FILES}
    ${RENDERERINFO_XML_FILES}
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/resources" PREFIX "resources" FILES ${RESOURCES_XML_FILES})
source_group("rendererInfo" FILES ${RENDERERINFO_XML_FILES})
set_source_files_properties(${RESOURCES_XML_FILES} PROPERTIES HEADER_FILE_ONLY ON)
set_source_files_properties(${RENDERERINFO_XML_FILES} PROPERTIES HEADER_FILE_ONLY ON)

#--------------------------------------------------------------------------------------
