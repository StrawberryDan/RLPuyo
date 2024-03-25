cmake_minimum_required(VERSION 3.28)
project(RLPuyo)
include(FetchContent)


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

set(CMAKE_CXX_STANDARD 20)

add_executable(RLPuyo src/main.cpp)
target_link_libraries(RLPuyo PUBLIC StrawberryCore StrawberryWindow StrawberryVulkan)