cmake_minimum_required(VERSION 2.8)

add_subdirectory(camera)
add_subdirectory(preprocessing)
add_subdirectory(detection)
add_subdirectory(tracking)

add_library(DropletTracking_LIB droplet_tracking.cpp)
target_link_libraries(DropletTracking_LIB ${OpenCV_LIBS} Camera_LIB PreProcessing_LIB Detection_LIB Tracking_LIB)