cmake_minimum_required(VERSION 2.8)

add_library(PreProcessing_LIB STATIC preprocessing.cpp perspective_transform.cpp segment.cpp)
target_link_libraries(PreProcessing_LIB ${OpenCV_LIBS})