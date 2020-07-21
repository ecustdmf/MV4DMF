cmake_minimum_required(VERSION 2.8)

add_library(Camera_LIB STATIC camera.cpp)
target_link_libraries(Camera_LIB ${OpenCV_LIBS})