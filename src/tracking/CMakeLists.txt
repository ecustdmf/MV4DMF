cmake_minimum_required(VERSION 2.8)

add_subdirectory("CF Tracker")

add_library(Tracking_LIB STATIC tracking.cpp)
target_link_libraries(Tracking_LIB ${OpenCV_LIBS} CFT_LIB)