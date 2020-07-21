cmake_minimum_required(VERSION 2.8)

set(PROJ DropletTracking)
set(CMAKE_CXX_STANDARD 11)
project(${PROJ})

set(OpenCV_DIR "opencv/MinGWbuild")
find_package(OpenCV 4 REQUIRED)

add_subdirectory(src)

add_executable(${PROJ} main.cpp)
target_link_libraries(${PROJ} ${OpenCV_LIBS} DropletTracking_LIB)