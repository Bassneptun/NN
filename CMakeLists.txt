cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(NN)
set(NN_VERSION 1.0)

add_compile_options(--debug)

find_package(Armadillo REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main classical.cpp)
target_link_libraries(main PRIVATE armadillo)
