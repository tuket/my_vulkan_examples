cmake_minimum_required(VERSION 3.23) # TODO: change to 3.24 once it is released
project(vulkan_examples)

set (CMAKE_CXX_STANDARD 20)

#find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)
find_package(Vulkan REQUIRED)
find_library(SHADERC_LIBRARIES shaderc_combined $ENV{VULKAN_SDK}/lib)

## ShaderC
find_path(shaderc_INCLUDE_DIR
    NAMES shaderc/shaderc.h
    HINTS "$ENV{VULKAN_SDK}/include")
find_library(shaderc_LIBRARY
    NAMES shaderc_combined
    HINTS "$ENV{VULKAN_SDK}/lib")

add_library(shaderc STATIC IMPORTED)
set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION_RELEASE "$ENV{VULKAN_SDK}/lib/shaderc_combined.lib")
set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION_DEBUG "$ENV{VULKAN_SDK}/lib/shaderc_combinedd.lib")
set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION_MINSIZEREL "$ENV{VULKAN_SDK}/lib/shaderc_combined.lib")
set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "$ENV{VULKAN_SDK}/lib/shaderc_combined.lib")
set_target_properties(shaderc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${shaderc_INCLUDE_DIR}")

add_subdirectory(libs/glm)
add_subdirectory(libs/glfw)

add_executable(vulkan_examples
	src/main.cpp
)
#target_link_libraries(vulkan_examples Vulkan::Vulkan Vulkan::shaderc_combined glm glfw)
target_link_libraries(vulkan_examples Vulkan::Vulkan ${SHADERC_LIBRARIES} glm glfw)
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT vulkan_examples)
set_target_properties(
    vulkan_examples PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
