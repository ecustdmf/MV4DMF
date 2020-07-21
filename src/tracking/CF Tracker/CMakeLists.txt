cmake_minimum_required(VERSION 2.8)

add_library(CFT_LIB STATIC KCF_Tracker.cpp ColorName_Feature.cpp)
target_link_libraries(CFT_LIB ${OpenCV_LIBS})