cmake_minimum_required(VERSION 3.28)
project(RLPuyo)
include(FetchContent)


if (PROJECT_IS_TOP_LEVEL)
    FetchContent_Declare(StrawberryConfig
            GIT_REPOSITORY "https://github.com/StrawberryDan/Config.git"
            GIT_TAG "main")
    FetchContent_MakeAvailable(StrawberryConfig)
    include(StrawberryConfig)
endif()


fetchcontent_declare(StrawberryCore
    GIT_REPOSITORY https://github.com/StrawberryDan/Core.git
    GIT_TAG main)
fetchcontent_declare(StrawberryWindow
    GIT_REPOSITORY https://github.com/StrawberryDan/Window.git
    GIT_TAG main)
fetchcontent_declare(StrawberryVulkan
    GIT_REPOSITORY https://github.com/StrawberryDan/Vulkan.git
    GIT_TAG main)
fetchcontent_makeavailable(StrawberryCore StrawberryWindow StrawberryVulkan)

add_executable(RLPuyo
        src/main.cpp
        src/Game/Board.hpp
        src/Game/Board.cpp
        src/Game/Renderer.hpp
        src/Game/Renderer.cpp)
set_target_properties(RLPuyo PROPERTIES CXX_STANDARD 20)
target_link_libraries(RLPuyo PUBLIC StrawberryCore StrawberryWindow StrawberryVulkan)
add_target_shaders(TARGET RLPuyo SHADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Game/Board.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Game/Board.frag
)


if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    target_link_options(RLPuyo PRIVATE "-static")
endif()


set(RLPUYO_REALTIME OFF CACHE BOOL "Whether the game should operate on human-scale timings")
if (RLPUYO_REALTIME)
    target_compile_definitions(RLPuyo PUBLIC RLPUYO_REALTIME=1)
else()
    target_compile_definitions(RLPuyo PUBLIC RLPUYO_REALTIME=0)
endif()