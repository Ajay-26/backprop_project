cmake_minimum_required(VERSION 3.15)

project(HPC-ML)

include_directories(
    inc/dataset
    inc/model
    )

file(GLOB SOURCES src/*/*.cpp)
set(SOURCES ${SOURCES} main.cpp)

message(STATUS "value of source is ${SOURCES}")
add_executable(hpc ${SOURCES})
