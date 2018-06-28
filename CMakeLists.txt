cmake_minimum_required(VERSION 3.10)
project(FinalProjectCV)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(FinalProjectCV main.cpp peopleCounter.cpp peopleCounter.h)
target_link_libraries(FinalProjectCV ${OpenCV_LIBS})
