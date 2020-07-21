#pragma once
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace camera {
    void open(int cameraIndex = 0, double _width = -1, double _height = -1, double _fps = 30);
    void setParams(double brightness, double contrast, double saturation, double hue, double gain, double exposure);
    static void get_webcam_fps();
    void openVideo(char* path);
    bool read(cv::Mat&);
    void close();
}


namespace recorder {
    void open(cv::Size size = cv::Size(-1, -1));
    void record(cv::Mat&);
    void close();
}