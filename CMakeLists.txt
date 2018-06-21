cmake_minimum_required(VERSION 3.10)
project(FinalProjectCV)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${FinalProjectCV_SOURCE_DIR}/opencv_contrib-3.4.1")

add_executable(FinalProjectCV main.cpp peopleCounter.cpp peopleCounter.h TrackbarsManager.cpp TrackbarsManager.h)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(FinalProjectCV ${OpenCV_LIBS})
