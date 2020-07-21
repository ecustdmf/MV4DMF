cmake_minimum_required(VERSION 2.8)

add_library(Detection_LIB STATIC detect.cpp)
target_link_libraries(Detection_LIB ${OpenCV_LIBS})